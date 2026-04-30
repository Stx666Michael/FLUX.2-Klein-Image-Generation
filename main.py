import argparse
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import get_token

# ---------------------------------------------------------------------------
# Available models — Klein models use Flux2KleinPipeline with a built-in
# text encoder and run fully on-device.
# ---------------------------------------------------------------------------
MODELS = {
    "flux2-klein-4b": {"repo": "black-forest-labs/FLUX.2-klein-4B"},
    "flux2-klein-9b": {"repo": "black-forest-labs/FLUX.2-klein-9B"},
}

DEFAULTS = {"steps": 4, "guidance": 1.0, "dtype": torch.float16}

device = "mps"


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein image generation (Apple Silicon)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="flux2-klein-4b",
        help="Model to use. Choices: " + ", ".join(MODELS.keys()),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Realistic macro photograph of a hermit crab using a soda can as its shell, "
            "partially emerging from the can, captured with sharp detail and natural colors, "
            "on a sunlit beach with soft shadows and a shallow depth of field, with blurred "
            "ocean waves in the background. The can has the text `BFL Diffusers` on it and "
            "it has a color gradient that start with #FF5733 at the top and transitions to "
            "#33FF57 at the bottom."
        ),
    )
    parser.add_argument("--steps",    type=int,   default=None,  help="Number of inference steps (default: 4)")
    parser.add_argument("--guidance", type=float, default=None,  help="Guidance scale (default: 1.0)")
    parser.add_argument("--seed",     type=int,   default=42,    help="Random seed")
    parser.add_argument("--size",     type=int,   default=1024,  help="Image size in pixels (width and height, default: 1024)")
    parser.add_argument("--output",   type=str,   default=None,  help="Output file path (default: <model>.png)")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_id = MODELS[args.model]["repo"]
    steps    = args.steps    if args.steps    is not None else DEFAULTS["steps"]
    guidance = args.guidance if args.guidance is not None else DEFAULTS["guidance"]
    output_path = args.output or f"{args.model}.png"

    print(f"Model  : {args.model} ({repo_id})")
    print(f"Steps  : {steps}  |  Guidance: {guidance}  |  Seed: {args.seed}  |  Size: {args.size}x{args.size}")
    print(f"Output : {output_path}")

    pipe = Flux2KleinPipeline.from_pretrained(
        repo_id, torch_dtype=DEFAULTS["dtype"], token=get_token()
    )
    pipe.enable_model_cpu_offload()  # offload layers to CPU to fit in 24 GB unified RAM

    image = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]

    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
