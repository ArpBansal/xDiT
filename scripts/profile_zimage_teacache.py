"""
Profile ZImage to calibrate Tea Cache polynomial coefficients.

Run (single GPU, no torchrun needed):
    python scripts/profile_zimage_teacache.py \
        --model Tongyi-MAI/Z-Image \
        --num_inference_steps 30 \
        --num_prompts 4

This script:
1. Runs inference WITHOUT cache, hooking into each ZImageTransformerBlock
   to record the modulated input (attention_norm1(x) * (1+scale_msa)) at every step.
2. Computes rel-L1 distances between consecutive steps for `layers[0]`.
3. Records whether the full-pass output residual changed significantly (ground truth).
4. Fits a polynomial of chosen degree that maps raw L1 → an accumulated score,
   such that the accumulated score separates "can skip" from "must compute".
"""

import argparse
import torch
import numpy as np
from diffusers import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--num_inference_steps", type=int, default=30)
parser.add_argument("--num_prompts", type=int, default=4,
                    help="Number of random prompts to average over")
parser.add_argument("--poly_degree", type=int, default=4,
                    help="Degree of polynomial to fit")
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
args = parser.parse_args()

PROMPTS = [
    "a red cat sitting on a bench",
    "a futuristic city at night with neon lights",
    "a portrait of an astronaut on the moon",
    "a serene mountain lake at sunrise",
][:args.num_prompts]

# --------------------------------------------------------------------------- #
# Load model
# --------------------------------------------------------------------------- #
print(f"Loading {args.model} ...")
pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
transformer = pipe.transformer

# --------------------------------------------------------------------------- #
# Hook: capture modulated input of layers[0] at each step
# --------------------------------------------------------------------------- #
# layers[0] has modulation=True; signal = attention_norm1(x) * (1 + scale_msa)
# shape: [B, S, D] — we take the mean over B,S,D for comparisons

_step_modulated = []   # list of tensors, one per step
_step_output    = []   # output of layers[0] per step, to compute output residuals

def _make_hook(block: ZImageTransformerBlock):
    def forward_hook(module, args_, output):
        x          = args_[0]                 # (B, S, D)
        adaln_input = args_[3] if len(args_) > 3 else None

        if adaln_input is not None and module.modulation:
            mod = module.adaLN_modulation(adaln_input)          # (B, 4*D  small)
            scale_msa, _, _, _ = mod.unsqueeze(1).chunk(4, dim=2)
            modulated = module.attention_norm1(x) * (1.0 + scale_msa)
        else:
            modulated = module.attention_norm1(x)

        _step_modulated.append(modulated.detach().float().cpu())
        _step_output.append(output.detach().float().cpu())
    return forward_hook

hook_handle = transformer.layers[0].register_forward_hook(_make_hook(transformer.layers[0]))

# --------------------------------------------------------------------------- #
# Collect data across prompts
# --------------------------------------------------------------------------- #
all_l1_distances = []     # rel-L1(modulated_t, modulated_{t-1})
all_output_changes = []   # rel-L1(output_t, output_{t-1})  — the "ground truth"

for prompt in PROMPTS:
    _step_modulated.clear()
    _step_output.clear()

    with torch.no_grad():
        pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            output_type="latent",
        )

    n = len(_step_modulated)
    print(f"Prompt '{prompt[:40]}': captured {n} steps from layers[0]")

    for t in range(1, n):
        m_prev = _step_modulated[t - 1]
        m_curr = _step_modulated[t]
        o_prev = _step_output[t - 1]
        o_curr = _step_output[t]

        # relative L1 of modulated inputs (the cache signal)
        l1 = ((m_curr - m_prev).abs().mean() / (m_prev.abs().mean() + 1e-8)).item()

        # relative L1 of outputs (surrogate for "how different is this step's contribution")
        out_change = ((o_curr - o_prev).abs().mean() / (o_prev.abs().mean() + 1e-8)).item()

        all_l1_distances.append(l1)
        all_output_changes.append(out_change)

hook_handle.remove()

# --------------------------------------------------------------------------- #
# Fit polynomial:  output_change ≈ poly(l1_distance)
# --------------------------------------------------------------------------- #
x = np.array(all_l1_distances,  dtype=np.float64)
y = np.array(all_output_changes, dtype=np.float64)

coeffs = np.polyfit(x, y, deg=args.poly_degree)
poly   = np.poly1d(coeffs)
y_pred = poly(x)
residuals = y - y_pred
r2 = 1.0 - (residuals.var() / (y.var() + 1e-12))

print("\n" + "="*60)
print(f"Polynomial degree: {args.poly_degree}")
print(f"R² on collected data: {r2:.4f}")
print(f"\nCoefficients (highest degree first):\n  {list(coeffs)}")
print("\nPaste into xfuser/model_executor/cache/utils.py CacheContext.__init__:")
coeff_str = ", ".join(f"{c:.8f}" for c in coeffs)
print(f'  self.register_buffer("z_image_coef",')
print(f'      torch.tensor([{coeff_str}]).to(get_device(0)))')

# --------------------------------------------------------------------------- #
# Diagnostic plot (optional, skipped if matplotlib not available)
# --------------------------------------------------------------------------- #
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(x, y, alpha=0.5, s=10, label="data")
    xs = np.linspace(x.min(), x.max(), 200)
    axes[0].plot(xs, poly(xs), "r-", label=f"poly deg={args.poly_degree}")
    axes[0].set_xlabel("rel-L1(modulated_t, modulated_{t-1})")
    axes[0].set_ylabel("rel-L1(output_t, output_{t-1})")
    axes[0].set_title("Fit quality")
    axes[0].legend()

    axes[1].plot(range(len(x)), x, label="input l1")
    axes[1].plot(range(len(y)), y, label="output change")
    axes[1].set_xlabel("step index (all prompts concatenated)")
    axes[1].set_title("L1 distances over time")
    axes[1].legend()

    plt.tight_layout()
    out_path = "zimage_teacache_profile.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved to {out_path}")
except ImportError:
    print("\n(matplotlib not available, skipping plot)")
