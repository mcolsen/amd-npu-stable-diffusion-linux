#!/usr/bin/env python3
"""Smoke test: load the SD1.5 UNet replaced.onnx with DD custom ops and run one
inference with random inputs. Exits 0 on success.

Uses local ./lib/ and ./models/ paths from ./setup.sh — run that first. Honors
XDNA_MODELS_DIR for non-default model parent directories.

Usage:
    source .venv/bin/activate
    python tests/test_dd_smoke.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TEST_DIR)

CUSTOM_OP_LIB = os.path.join(_PROJECT_DIR, "lib", "libonnx_custom_ops.so")
DD_LIB_DIR = os.path.join(_PROJECT_DIR, "lib")

_models_root = os.environ.get("XDNA_MODELS_DIR", os.path.join(_PROJECT_DIR, "models"))
MODEL_DIR = os.path.join(_models_root, "stable-diffusion-1.5-amdnpu")
UNET_ONNX = os.path.join(MODEL_DIR, "unet", "dd", "replaced.onnx")

# Ensure DD runtime libraries are findable before onnxruntime imports.
os.environ["DD_ROOT"] = os.environ.get("DD_ROOT", DD_LIB_DIR)
if DD_LIB_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = DD_LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import onnxruntime  # noqa: E402


def main() -> None:
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"Available providers:  {onnxruntime.get_available_providers()}")
    print()

    if not os.path.exists(CUSTOM_OP_LIB):
        sys.exit(f"FAIL: custom ops library not found: {CUSTOM_OP_LIB}\n"
                 "  Run ./setup.sh first.")
    if not os.path.exists(UNET_ONNX):
        sys.exit(f"FAIL: UNet ONNX not found: {UNET_ONNX}\n"
                 "  Run: python download_model.py --model sd15")

    print(f"Loading custom ops: {CUSTOM_OP_LIB}")
    ctypes.CDLL(CUSTOM_OP_LIB)

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    cache_dir = os.path.join(os.path.dirname(UNET_ONNX), ".cache")
    session_options.add_session_config_entry("dd_cache", cache_dir)
    session_options.add_session_config_entry("onnx_custom_ops_const_key", UNET_ONNX)
    session_options.register_custom_ops_library(CUSTOM_OP_LIB)

    print(f"Creating InferenceSession for: {UNET_ONNX}")
    t0 = time.perf_counter()
    session = onnxruntime.InferenceSession(
        UNET_ONNX,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    print(f"  session created in {time.perf_counter() - t0:.2f}s\n")

    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")
    print()

    # Feed random inputs matching the declared shapes/dtypes.
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str):
                # Dynamic dim — pick SD1.5 defaults (batch=2 for CFG, CLIP seq=77, latent=64).
                dim_lc = dim.lower()
                if "batch" in dim_lc:
                    shape.append(2)
                elif "sequence" in dim_lc or "length" in dim_lc:
                    shape.append(77)
                else:
                    shape.append(64)
            else:
                shape.append(dim)
        if "float16" in inp.type.lower() or "bfloat" in inp.type.lower():
            dtype = np.float16
        elif "double" in inp.type.lower() or "float64" in inp.type.lower():
            dtype = np.float64
        elif "int64" in inp.type.lower():
            dtype = np.int64
        elif "int32" in inp.type.lower():
            dtype = np.int32
        else:
            dtype = np.float32
        feeds[inp.name] = np.random.randn(*shape).astype(dtype)
        print(f"  input '{inp.name}': {feeds[inp.name].shape} ({feeds[inp.name].dtype})")
    print()

    print("Running inference...")
    output_names = [out.name for out in session.get_outputs()]
    t0 = time.perf_counter()
    results = session.run(output_names, feeds)
    print(f"  inference completed in {time.perf_counter() - t0:.2f}s\n")

    for name, result in zip(output_names, results):
        print(f"  output '{name}': shape={result.shape}, dtype={result.dtype}")
        print(f"    min={result.min():.6f} max={result.max():.6f} mean={result.mean():.6f}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
