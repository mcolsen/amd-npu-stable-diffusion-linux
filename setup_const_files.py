#!/usr/bin/env python3
"""Extract individual .const files from packed .fconst for UNet and VAE decoder.

AMD's pre-compiled ONNX models ship .fconst files (packed constants) but the DD
runtime needs individual .const files referenced by the meta.json. This script
extracts them using offset/size info from meta.json.

Run once after downloading a model:

    python setup_const_files.py --model-dir models/stable-diffusion-1.5-amdnpu
    python setup_const_files.py --model-dir models/segmind-vega-amdnpu

Or, more commonly, called transitively by download_model.py / setup.sh.
"""

import argparse
import json
import os
import sys

# SD 1.5 and SD-Turbo share the same fconst key names.
# SDXL uses `unetconv_inConv` for the UNet entry (not `conv_inConv`);
# its VAE decoder still uses the `post_quant_convConv` naming.
_SD15_COMPONENTS = [
    {
        "name": "UNet",
        "meta": "unet/dd/cache/NhwcConv_0-conv_inConv_meta.json",
        "fconst": "unet/dd/dd_metastate_SD15_Unet_NhwcConv_0-conv_inConv.fconst",
        "base_dir": "unet/dd",
    },
    {
        "name": "VAE Decoder",
        "meta": "vae_decoder/dd/cache/NhwcConv_0-post_quant_convConv_meta.json",
        "fconst": "vae_decoder/dd/dd_metastate_Sd15_Decoder_NhwcConv_0-post_quant_convConv.fconst",
        "base_dir": "vae_decoder/dd",
    },
]

_SDXL_COMPONENTS = [
    {
        "name": "UNet",
        "meta": "unet/dd/cache/NhwcConv_0-unetconv_inConv_meta.json",
        "fconst": "unet/dd/dd_metastate_SD15_Unet_NhwcConv_0-unetconv_inConv.fconst",
        "base_dir": "unet/dd",
    },
    {
        "name": "VAE Decoder",
        "meta": "vae_decoder/dd/cache/NhwcConv_0-post_quant_convConv_meta.json",
        "fconst": "vae_decoder/dd/dd_metastate_Sd15_Decoder_NhwcConv_0-post_quant_convConv.fconst",
        "base_dir": "vae_decoder/dd",
    },
]


def components_for(model_dir):
    basename = os.path.basename(os.path.normpath(model_dir))
    # Segmind Vega is a distilled SDXL and ships the same fconst key names.
    if "sdxl" in basename or "vega" in basename:
        return _SDXL_COMPONENTS
    return _SD15_COMPONENTS


def extract_consts(component, model_dir):
    meta_path = os.path.join(model_dir, component["meta"])
    fconst_path = os.path.join(model_dir, component["fconst"])
    base_dir = os.path.join(model_dir, component["base_dir"])

    if not os.path.exists(meta_path):
        print(f"  SKIP - meta.json not found: {meta_path}")
        return 0

    if not os.path.exists(fconst_path):
        print(f"  SKIP - fconst not found: {fconst_path}")
        return 0

    with open(meta_path) as f:
        meta = json.load(f)

    with open(fconst_path, "rb") as f:
        fconst_data = f.read()

    os.makedirs(os.path.join(base_dir, ".cache"), exist_ok=True)

    count = 0
    for name, info in meta["tensor_map"].items():
        if "file_name" not in info:
            continue

        offset = info["offset"]
        size = info["file_size"]
        fname = info["file_name"]
        out_path = os.path.join(base_dir, fname)

        if os.path.exists(out_path) and os.path.getsize(out_path) == size:
            count += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if offset + size <= len(fconst_data):
            with open(out_path, "wb") as f:
                f.write(fconst_data[offset : offset + size])
        else:
            # Last entry may extend past fconst; pad with zeros
            available = len(fconst_data) - offset
            with open(out_path, "wb") as f:
                f.write(fconst_data[offset:])
                f.write(b"\x00" * (size - available))

        count += 1

    return count


def extract_all(model_dir):
    """Extract .const files for every component in a model directory.

    Returns the total count of const files present (extracted or already-cached).
    Raises FileNotFoundError if the model directory itself doesn't exist.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    total = 0
    for component in components_for(model_dir):
        print(f"Extracting {component['name']} const files...")
        count = extract_consts(component, model_dir)
        print(f"  {count} const files ready")
        total += count
    return total


def main():
    parser = argparse.ArgumentParser(description="Extract .const files from .fconst")
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to the downloaded model directory (e.g. models/stable-diffusion-1.5-amdnpu)",
    )
    args = parser.parse_args()

    try:
        extract_all(args.model_dir)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Model directory: {args.model_dir}")


if __name__ == "__main__":
    main()
