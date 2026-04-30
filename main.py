import argparse

from generator import DEFAULTS, MODELS, generate


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
    parser.add_argument("--seed",     type=int,   default=DEFAULTS["seed"], help="Random seed")
    parser.add_argument("--width",    type=int,   default=DEFAULTS["width"],  help="Output image width in pixels (default: 1024)")
    parser.add_argument("--height",   type=int,   default=DEFAULTS["height"], help="Output image height in pixels (default: 1024)")
    parser.add_argument("--image",    type=str,   nargs="+",     help="One or more input images for editing (local path or URL)")
    parser.add_argument("--output",   type=str,   default=None,  help="Output file path (default: <model>.png)")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_id = MODELS[args.model]["repo"]
    steps    = args.steps    if args.steps    is not None else DEFAULTS["steps"]
    guidance = args.guidance if args.guidance is not None else DEFAULTS["guidance"]
    output_path = args.output or f"{args.model}.png"
    mode = "edit" if args.image else "generate"

    print(f"Model  : {args.model} ({repo_id})")
    print(f"Mode   : {mode}{f' ({len(args.image)} input image(s))' if args.image else ''}")
    print(f"Steps  : {steps}  |  Guidance: {guidance}  |  Seed: {args.seed}  |  Size: {args.width}x{args.height}")
    print(f"Output : {output_path}")

    image = generate(
        model=args.model,
        prompt=args.prompt,
        images=args.image,
        steps=steps,
        guidance=guidance,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )

    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
