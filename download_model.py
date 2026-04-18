#!/usr/bin/env python3
"""Download a pre-compiled Stable Diffusion model from Hugging Face and prepare
it for the NPU runtime.

For each preset this:
  1. Downloads the AMD-published ONNX model from Hugging Face into ./models/<dir>/
  2. Unpacks the .fconst archives into individual .const files that the DD
     runtime expects.

Four of the five presets are gated: you must request access on the model page
and run `huggingface-cli login` first. Segmind Vega is public and requires
neither.

Usage:
    python download_model.py --model sd15
    python download_model.py --model vega --models-dir /shared/models

Respects the XDNA_MODELS_DIR environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys

import setup_const_files

# Preset -> (HF repo, local directory name, gated?).
# Local directory names match those used by run_npu.py's MODELS table.
MODELS = {
    "sd15":       ("amd/stable-diffusion-1.5-amdnpu", "stable-diffusion-1.5-amdnpu", True),
    "sd-turbo":   ("amd/sd-turbo-amdnpu",             "sd-turbo-amdnpu",             True),
    "sdxl-base":  ("amd/sdxl-base-amdnpu",            "sdxl-base-amdnpu",            True),
    "sdxl-turbo": ("amd/sdxl-turbo-amdnpu",           "sdxl-turbo-amdnpu",           True),
    "vega":       ("amd/segmind-vega-amdnpu",         "segmind-vega-amdnpu",         False),
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_models_dir(override: str | None) -> str:
    if override:
        return override
    env = os.environ.get("XDNA_MODELS_DIR")
    if env:
        return env
    return os.path.join(_SCRIPT_DIR, "models")


def ensure_downloaded(repo: str, local_dir: str, gated: bool) -> None:
    sentinel = os.path.join(local_dir, "unet", "dd", "replaced.onnx")
    if os.path.exists(sentinel):
        print(f"  Already downloaded: {local_dir}")
        return

    print(f"  Downloading {repo} -> {local_dir}")
    if gated:
        print("  (gated model — if this fails with 401, request access at "
              f"https://huggingface.co/{repo} and run `huggingface-cli login`)")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run ./setup.sh first, "
              "or `pip install huggingface_hub` inside your venv.", file=sys.stderr)
        sys.exit(1)

    try:
        snapshot_download(repo, local_dir=local_dir, local_dir_use_symlinks=False)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "401" in msg or "gated" in msg.lower() or "authentication" in msg.lower():
            print(f"\nERROR: Hugging Face denied access to {repo}.", file=sys.stderr)
            print("  1. Visit the model page and click 'Request access':", file=sys.stderr)
            print(f"     https://huggingface.co/{repo}", file=sys.stderr)
            print("  2. Once approved, authenticate:", file=sys.stderr)
            print("       huggingface-cli login", file=sys.stderr)
            sys.exit(1)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pre-compiled SD model from Hugging Face")
    parser.add_argument(
        "--model", required=True, choices=sorted(MODELS.keys()),
        help="Which preset to download",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Parent directory for model checkouts (default: ./models, or $XDNA_MODELS_DIR)",
    )
    args = parser.parse_args()

    repo, dir_name, gated = MODELS[args.model]
    models_root = resolve_models_dir(args.models_dir)
    os.makedirs(models_root, exist_ok=True)
    local_dir = os.path.join(models_root, dir_name)

    print(f"Model: {args.model}")
    print(f"  HF repo: {repo}")
    print(f"  Local:   {local_dir}")
    print()

    ensure_downloaded(repo, local_dir, gated)
    print()
    print("Extracting .const files...")
    setup_const_files.extract_all(local_dir)
    print()
    print(f"Done. Run: ./run.sh --model {args.model} \"your prompt\"")


if __name__ == "__main__":
    main()
