import os
import io
import time
import base64
import logging
import argparse
from typing import Optional

import torch
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import T5EncoderModel

from xfuser import xFuserArgs
from xfuser.model_executor.pipelines.pipeline_flux_control import xFuserPipelineFluxControlPipeline


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    # control_image must be provided as a base64-encoded PNG/JPEG string
    control_image_b64: str
    num_inference_steps: Optional[int] = 28
    seed: Optional[int] = 42
    guidance_scale: Optional[float] = 3.5
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    max_sequence_length: Optional[int] = 512
    # Where to persist the result; if None the PNG is returned as base64
    save_disk_path: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a robot in a futuristic city",
                "control_image_b64": "<base64-encoded PNG>",
                "num_inference_steps": 28,
                "seed": 42,
                "guidance_scale": 3.5,
                "height": 1024,
                "width": 1024,
                "max_sequence_length": 512,
            }
        }


# ---------------------------------------------------------------------------
# Ray worker
# ---------------------------------------------------------------------------

@ray.remote(num_gpus=1)
class FluxControlWorker:
    def __init__(
        self,
        xfuser_args: xFuserArgs,
        rank: int,
        world_size: int,
    ):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        self.rank = rank
        self._setup_logger()
        self._init_pipeline(xfuser_args)

    def _setup_logger(self):
        self.logger = logging.getLogger(f"FluxControlWorker-{self.rank}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _init_pipeline(self, xfuser_args: xFuserArgs):
        self.engine_config, self.input_config = xfuser_args.create_config()
        dtype = torch.bfloat16

        # Load T5 text encoder (optionally quantised to FP8)
        self.logger.info("Loading T5 text encoder …")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            self.engine_config.model_config.model,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        )

        if xfuser_args.use_fp8_t5_encoder:
            try:
                from optimum.quanto import freeze, qfloat8, quantize
                self.logger.info(f"Rank {self.rank}: quantising T5 to FP8 …")
                quantize(text_encoder_2, weights=qfloat8)
                freeze(text_encoder_2)
            except ImportError:
                self.logger.warning(
                    "optimum-quanto not installed – skipping FP8 quantisation. "
                    "Install it with: pip install optimum-quanto"
                )

        cache_args = {
            "use_teacache": xfuser_args.use_teacache,
            "use_fbcache": xfuser_args.use_fbcache,
            "rel_l1_thresh": 0.12,
            "return_hidden_states_first": False,
            "num_steps": self.input_config.num_inference_steps,
        }

        self.logger.info("Loading Flux Control pipeline …")
        self.pipe = xFuserPipelineFluxControlPipeline.from_pretrained(
            pretrained_model_name_or_path=self.engine_config.model_config.model,
            engine_config=self.engine_config,
            cache_args=cache_args,
            torch_dtype=dtype,
            text_encoder_2=text_encoder_2,
        ).to(f"cuda:{self.rank}")

        self.pipe.prepare_run(self.input_config, steps=self.input_config.num_inference_steps)
        self.logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, request: GenerateRequest):
        # Decode control image from base64
        try:
            img_bytes = base64.b64decode(request.control_image_b64)
            control_image = (
                Image.open(io.BytesIO(img_bytes))
                .convert("RGB")
                .resize((request.width, request.height))
            )
        except Exception as exc:
            raise ValueError(f"Failed to decode control_image_b64: {exc}") from exc

        t0 = time.time()
        output = self.pipe(
            height=request.height,
            width=request.width,
            prompt=request.prompt,
            control_image=control_image,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            max_sequence_length=request.max_sequence_length,
            output_type="pil",
            generator=torch.Generator(device="cuda").manual_seed(request.seed),
        )
        elapsed = time.time() - t0

        if not self.pipe.is_dp_last_group():
            return None

        image = output.images[0]

        if request.save_disk_path:
            os.makedirs(request.save_disk_path, exist_ok=True)
            filename = f"flux_control_{time.strftime('%Y%m%d-%H%M%S')}.png"
            path = os.path.join(request.save_disk_path, filename)
            image.save(path)
            return {
                "message": "Image generated successfully",
                "elapsed_time": f"{elapsed:.2f}s",
                "output": path,
                "save_to_disk": True,
            }
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            return {
                "message": "Image generated successfully",
                "elapsed_time": f"{elapsed:.2f}s",
                "output": img_b64,
                "save_to_disk": False,
            }


# ---------------------------------------------------------------------------
# Engine – manages the Ray worker pool
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, world_size: int, xfuser_args: xFuserArgs):
        if not ray.is_initialized():
            ray.init()

        self.workers = [
            FluxControlWorker.remote(xfuser_args, rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]

    async def generate(self, request: GenerateRequest):
        results = ray.get([w.generate.remote(request) for w in self.workers])
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise RuntimeError("No worker returned a result.")
        return result


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="xDiT Flux Control Service")


@app.post("/generate")
async def generate_image(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    if not request.control_image_b64:
        raise HTTPException(status_code=400, detail="control_image_b64 cannot be empty")
    if request.height <= 0 or request.width <= 0:
        raise HTTPException(status_code=400, detail="height and width must be positive")
    if request.num_inference_steps <= 0:
        raise HTTPException(status_code=400, detail="num_inference_steps must be positive")

    try:
        return await engine.generate(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xDiT Flux Control HTTP Service")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path or local directory")
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs / Ray workers")
    parser.add_argument("--pipefusion_parallel_degree", type=int, default=1)
    parser.add_argument("--ulysses_parallel_degree", type=int, default=2)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--use_cfg_parallel", action="store_true")
    parser.add_argument("--use_fp8_t5_encoder", action="store_true", help="Quantise T5 encoder to FP8 (requires optimum-quanto)")
    parser.add_argument("--use_teacache", action="store_true", help="Enable TeaCache attention caching")
    parser.add_argument("--use_fbcache", action="store_true", help="Enable FBCache")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    cli = parser.parse_args()

    xfuser_args = xFuserArgs(
        model=cli.model_path,
        trust_remote_code=True,
        warmup_steps=cli.warmup_steps,
        use_parallel_vae=False,
        use_torch_compile=False,
        ulysses_degree=cli.ulysses_parallel_degree,
        ring_degree=cli.ring_degree,
        pipefusion_parallel_degree=cli.pipefusion_parallel_degree,
        use_cfg_parallel=cli.use_cfg_parallel,
        use_fp8_t5_encoder=cli.use_fp8_t5_encoder,
        use_teacache=cli.use_teacache,
        use_fbcache=cli.use_fbcache,
        dit_parallel_size=0,
    )

    engine = Engine(world_size=cli.world_size, xfuser_args=xfuser_args)

    import uvicorn
    uvicorn.run(app, host=cli.host, port=cli.port)
