"""Shared FLUX.2 Klein pipeline loading + generation.

Used by both the CLI (``main.py``) and the web UI (``app.py``) so the model
is loaded the same way in both contexts.
"""
from __future__ import annotations

import threading
from typing import Iterable, Optional

import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
from PIL import Image

# ---------------------------------------------------------------------------
# Available models — Klein models use Flux2KleinPipeline with a built-in
# text encoder and run fully on-device.
# ---------------------------------------------------------------------------
MODELS = {
    # Standard float16 variants (good for MPS / high-VRAM CUDA)
    "flux2-klein-4b": {"repo": "black-forest-labs/FLUX.2-klein-4B"},
    "flux2-klein-9b": {"repo": "black-forest-labs/FLUX.2-klein-9B"},
    # Quantized variants — significantly lower VRAM/RAM (CUDA only)
    "flux2-klein-4b-nf4": {"repo": "black-forest-labs/FLUX.2-klein-4B", "quantize": "nf4"},
    "flux2-klein-4b-int8": {"repo": "black-forest-labs/FLUX.2-klein-4B", "quantize": "int8"},
}

DEFAULTS = {"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024, "seed": 42, "dtype": torch.float16}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()

# Cache a single pipeline instance across calls (UI keeps it warm between
# messages; CLI uses it once per invocation).
_pipe = None
_pipe_key: Optional[str] = None
_pipe_lock = threading.Lock()


def _make_bnb_config(quantize: str):
    """Return a BitsAndBytesConfig for the requested quantization level.

    Requires ``bitsandbytes`` and a CUDA device.  ``quantize`` must be
    ``'nf4'`` (4-bit NormalFloat, ~2 GB for 4B) or ``'int8'`` (~4 GB for 4B).
    """
    try:
        from diffusers import BitsAndBytesConfig as BnbConfig
    except ImportError:
        from transformers import BitsAndBytesConfig as BnbConfig  # type: ignore[assignment]
    if quantize == "nf4":
        return BnbConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    if quantize == "int8":
        return BnbConfig(load_in_8bit=True)
    raise ValueError(f"Unknown quantize value: {quantize!r}")


def get_pipeline(model_key: str) -> Flux2KleinPipeline:
    """Return a cached pipeline for ``model_key``, loading/swapping if needed."""
    global _pipe, _pipe_key
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choices: {list(MODELS)}")
    model_cfg = MODELS[model_key]
    repo_id = model_cfg["repo"]
    # bitsandbytes requires CUDA; silently drop quantize on other devices.
    quantize = model_cfg.get("quantize") if DEVICE == "cuda" else None
    cache_key = f"{repo_id}:{quantize}"
    with _pipe_lock:
        if _pipe is None or _pipe_key != cache_key:
            # Drop any prior pipeline before loading a new one.
            _pipe = None
            _pipe_key = None
            load_kwargs: dict = {"torch_dtype": DEFAULTS["dtype"], "token": get_token()}
            if quantize:
                load_kwargs["quantization_config"] = _make_bnb_config(quantize)
            pipe = Flux2KleinPipeline.from_pretrained(repo_id, **load_kwargs)
            if quantize:
                # Quantized model is loaded directly onto the GPU by bitsandbytes;
                # CPU offload is incompatible with quantized layers.
                pipe.to(DEVICE)
            else:
                pipe.enable_model_cpu_offload()
            _pipe = pipe
            _pipe_key = cache_key
        return _pipe


_MAX_INPUT_DIM = 512


def _fit_image(img):
    """Resize *img* so its larger dimension is at most ``_MAX_INPUT_DIM`` px."""
    w, h = img.size
    if max(w, h) <= _MAX_INPUT_DIM:
        return img
    scale = _MAX_INPUT_DIM / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def generate(
    *,
    model: str,
    prompt: str,
    images: Optional[Iterable] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    seed: int = DEFAULTS["seed"],
    width: int = DEFAULTS["width"],
    height: int = DEFAULTS["height"],
    on_step=None,
):
    """Run the pipeline and return the first generated PIL image.

    ``images`` may be a list of PIL images, file paths, or URLs (anything
    accepted by ``diffusers.utils.load_image``), or ``None`` for text-to-image.
    ``on_step``, if provided, is called as ``on_step(step, total)`` after each
    denoising step so callers can track progress.
    """
    steps = steps if steps is not None else DEFAULTS["steps"]
    guidance = guidance if guidance is not None else DEFAULTS["guidance"]

    input_images = None
    if images:
        input_images = [img if hasattr(img, "size") else load_image(img) for img in images]
        input_images = [_fit_image(img) for img in input_images]

    def _step_callback(pipe, step_index, timestep, callback_kwargs):
        if on_step is not None:
            on_step(step_index + 1, steps)
        return callback_kwargs

    pipe = get_pipeline(model)
    with _pipe_lock:
        result = pipe(
            prompt=prompt,
            image=input_images,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
            callback_on_step_end=_step_callback,
        )
    return result.images[0]
