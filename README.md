# FLUX.2 Klein Image Generation

Local image generation using [FLUX.2 Klein](https://huggingface.co/black-forest-labs) models on Apple Silicon (MPS).

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 24 GB unified RAM recommended for the 9B model; 16 GB is sufficient for 4B

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Log in to HuggingFace (one-time):

```bash
huggingface-cli login
```

You must also accept the model license on HuggingFace before downloading:
- [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)

## Usage

```bash
python main.py [--model MODEL] [--prompt TEXT] [--size PX] [--steps N] [--guidance F] [--seed N] [--output PATH]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `flux2-klein-4b` | `flux2-klein-4b` or `flux2-klein-9b` |
| `--prompt` | *(hermit crab scene)* | Text prompt |
| `--size` | `1024` | Output image size in pixels (square) |
| `--steps` | `4` | Number of inference steps |
| `--guidance` | `1.0` | Guidance scale |
| `--seed` | `42` | Random seed for reproducibility |
| `--output` | `<model>.png` | Output file path |

### Examples

```bash
# Quick run with 4B model at 512px (faster, less memory)
python main.py --model flux2-klein-4b --size 512 --output output/result.png

# High-res with 9B model
python main.py --model flux2-klein-9b --size 1024 --output output/klein9b.png

# Custom prompt
python main.py --model flux2-klein-4b --prompt "A futuristic city at sunset, cinematic lighting"

# More inference steps for higher quality
python main.py --model flux2-klein-9b --steps 8 --size 768 --output output/hq.png
```

## Model Cache

Models are downloaded once and cached at `~/.cache/huggingface/hub/`. To store them on an external drive:

```bash
export HF_HOME=/path/to/external/drive/hf_cache
python main.py ...
```

Add the `export` line to `~/.zshrc` to make it permanent.
