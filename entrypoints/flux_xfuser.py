import os
import io
import time
import base64

import torch
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import T5EncoderModel

from xfuser import xFuserArgs, xFuserFluxPipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.ray.pipeline.pipeline_utils import RayDiffusionPipeline


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 28
    seed: Optional[int] = 42
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    guidance_scale: Optional[float] = 0.0
    max_sequence_length: Optional[int] = 256
    save_disk_path: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "num_inference_steps": 28,
                "seed": 42,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 0.0,
                "max_sequence_length": 256,
            }
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="xDiT Flux HTTP Service")


# ---------------------------------------------------------------------------
# Engine  –  thin wrapper around RayDiffusionPipeline (mirrors ray_flux_example.py)
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, engine_args: xFuserArgs):
        if not ray.is_initialized():
            ray.init()

        # Prevent create_config() from calling init_distributed_environment()
        # in the main process (FastAPI). With use_ray=True it skips that path
        # and uses ray_world_size as world_size instead.
        engine_args.use_ray = True
        engine_config, input_config = engine_args.create_config()
        engine_config.runtime_config.dtype = torch.bfloat16

        # Load T5 encoder inside each Ray worker (same pattern as ray_flux_example.py)
        encoder_kwargs = {
            "text_encoder_2": {
                "model_class": T5EncoderModel,
                "pretrained_model_name_or_path": engine_config.model_config.model,
                "subfolder": "text_encoder_2",
                "torch_dtype": torch.bfloat16,
            }
        }

        self.pipe = RayDiffusionPipeline.from_pretrained(
            PipelineClass=xFuserFluxPipeline,
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=torch.bfloat16,
            **encoder_kwargs,
        )
        self.pipe.prepare_run(input_config)

    async def generate(self, request: GenerateRequest):
        start_time = time.time()

        # Returns list[list[PIL.Image] | None] – same as ray_flux_example.py
        results = self.pipe(
            height=request.height,
            width=request.width,
            prompt=request.prompt,
            num_inference_steps=request.num_inference_steps,
            output_type="pil",
            max_sequence_length=request.max_sequence_length,
            guidance_scale=request.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(request.seed),
        )

        elapsed = time.time() - start_time

        # Only the dp-last-group worker returns images; others return None
        image = None
        for images in results:
            if images is not None:
                image = images[0]
                break

        if image is None:
            raise RuntimeError("No worker returned an image.")

        if request.save_disk_path:
            os.makedirs(request.save_disk_path, exist_ok=True)
            filename = f"generated_image_{time.strftime('%Y%m%d-%H%M%S')}.png"
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
# API route
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")
    if request.height <= 0 or request.width <= 0:
        raise HTTPException(status_code=400, detail="height and width must be positive")
    if request.num_inference_steps <= 0:
        raise HTTPException(status_code=400, detail="num_inference_steps must be positive")
    try:
        return await engine.generate(request)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reuse xFuser's own arg parser (same as ray_flux_example.py) so all
    # parallel / model / runtime flags are automatically available.
    parser = FlexibleArgumentParser(description="xDiT FLUX HTTP Service")
    xFuserArgs.add_cli_args(parser)
    # HTTP-specific extras
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)

    args = parser.parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine = Engine(engine_args=engine_args)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)