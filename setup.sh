#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a self-contained environment for running Stable Diffusion on
# the AMD XDNA2 NPU.
#
# What this does:
#   1. Locates the Ryzen AI SDK installation
#   2. Creates a local Python venv in .venv/
#   3. Copies the SDK's ONNX Runtime (with VitisAI EP) into the venv
#   4. Copies the minimum deployment libraries into lib/
#   5. Installs Python dependencies from requirements.txt
#   6. Downloads the selected pre-compiled model from Hugging Face
#   7. Extracts the .const files needed by the DD runtime
#
# Prerequisites (see README.md for the full list):
#   - AMD NPU driver stack: amdxdna DKMS module + XRT (from amd/xdna-driver)
#   - Ryzen AI SDK 1.7.x installed somewhere on the system
#   - Python matching the SDK's bundled ONNX Runtime wheel (3.12 in 1.7.x)
#   - Hugging Face account with access to the relevant amd/*-amdnpu repo
#
# Usage:
#   ./setup.sh                                 # install + download sd15 (default)
#   ./setup.sh --model vega                    # install + download Segmind Vega
#   RYZEN_AI_PATH=/opt/ryzen_ai ./setup.sh     # specify SDK location
#   NPUTOP_PATH=/path/to/nputop ./setup.sh     # local nputop clone instead of git install

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
LIB_DIR="$PROJECT_DIR/lib"

# --- Parse args ---
MODEL="sd15"
while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '3,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "=== xdna2-npu-diffusion setup (model: $MODEL) ==="
echo

# --- Step 1: Find Ryzen AI SDK ---
find_sdk() {
    if [ -n "${RYZEN_AI_PATH:-}" ] && [ -d "$RYZEN_AI_PATH" ]; then
        echo "$RYZEN_AI_PATH"
        return 0
    fi
    for candidate in \
        "$HOME/ryzen_ai-1.7.1" \
        /opt/ryzen_ai*; do
        if [ -d "$candidate/deployment/lib" ] && [ -f "$candidate/deployment/lib/libonnx_custom_ops.so" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

echo "Step 1: Locating Ryzen AI SDK..."
if SDK_PATH=$(find_sdk); then
    echo "  Found: $SDK_PATH"
else
    echo "  ERROR: Ryzen AI SDK not found." >&2
    echo "  Install it from https://ryzenai.docs.amd.com/ and either:" >&2
    echo "    - place it at ~/ryzen_ai-1.7.1 or /opt/ryzen_ai*, or" >&2
    echo "    - set RYZEN_AI_PATH=/path/to/sdk before running this script." >&2
    exit 1
fi
echo

# --- Step 2: Detect the Python version the SDK ships ORT for ---
# The Ryzen AI SDK bundles a custom-built onnxruntime wheel under
# lib/python3.X/site-packages/onnxruntime. Our venv must match that X.Y so the
# .so ABIs line up.
echo "Step 2: Detecting SDK Python version..."
SDK_ORT_DIR=$(find "$SDK_PATH/lib" -maxdepth 1 -type d -name 'python3.*' | head -n1 || true)
if [ -z "$SDK_ORT_DIR" ] || [ ! -d "$SDK_ORT_DIR/site-packages/onnxruntime" ]; then
    echo "  ERROR: could not find bundled ONNX Runtime in $SDK_PATH/lib/python3.*/site-packages/" >&2
    exit 1
fi
PYVER=$(basename "$SDK_ORT_DIR")   # e.g. python3.12
PYEXE="$PYVER"
if ! command -v "$PYEXE" >/dev/null 2>&1; then
    echo "  ERROR: $PYEXE not on PATH. The Ryzen AI SDK ships ONNX Runtime for $PYVER;" >&2
    echo "  install that Python version (on Fedora: sudo dnf install $PYVER) and re-run." >&2
    exit 1
fi
echo "  SDK uses $PYVER"
echo

# --- Step 3: Create venv ---
echo "Step 3: Creating Python venv..."
if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists at $VENV_DIR"
else
    "$PYEXE" -m venv "$VENV_DIR"
    echo "  Created $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "  Python: $(python --version)"
echo

# --- Step 4: Install Python dependencies ---
echo "Step 4: Installing Python dependencies..."
# PyTorch gets its own invocation because it needs the CPU-only index URL.
pip install --quiet "torch>=2.1,<3" --index-url https://download.pytorch.org/whl/cpu
# Everything else from requirements.txt.
pip install --quiet -r "$PROJECT_DIR/requirements.txt"

# nputop: prefer a local editable install when the user points at one; the
# requirements.txt line above will already have pulled it from git otherwise.
if [ -n "${NPUTOP_PATH:-}" ] && [ -d "$NPUTOP_PATH" ]; then
    echo "  Reinstalling nputop from local path: $NPUTOP_PATH"
    pip install --quiet -e "$NPUTOP_PATH"
fi
# Confirm it's importable; degrade gracefully if not.
if ! python -c "import nputop.ioctl" 2>/dev/null; then
    echo "  WARNING: nputop not importable; telemetry will run in sysfs-only mode." >&2
fi
echo "  Done"
echo

# --- Step 5: Copy ONNX Runtime from SDK ---
echo "Step 5: Copying ONNX Runtime from SDK..."
ORT_SRC="$SDK_ORT_DIR/site-packages/onnxruntime"
ORT_DST="$VENV_DIR/lib/$PYVER/site-packages/onnxruntime"
if [ -d "$ORT_DST" ] && python -c "import onnxruntime" 2>/dev/null; then
    echo "  ONNX Runtime already installed"
else
    rm -rf "$ORT_DST"
    cp -r "$ORT_SRC" "$ORT_DST"
    echo "  Copied $(du -sh "$ORT_DST" | cut -f1) from SDK"
fi
python -c "import onnxruntime; print(f'  Version: {onnxruntime.__version__}')"
echo

# --- Step 6: Copy deployment libraries ---
echo "Step 6: Copying deployment libraries..."
mkdir -p "$LIB_DIR"
DEPLOY_LIBS=(
    libonnx_custom_ops.so
    libdyn_dispatch_core.so
    libdyn_bins.so
    libryzen_mm.so
    libryzen_mm.so.1
    libryzen_mm.so.1.0.0
    libspdlog.so.1.15
)
copied=0
for lib in "${DEPLOY_LIBS[@]}"; do
    src="$SDK_PATH/deployment/lib/$lib"
    dst="$LIB_DIR/$lib"
    if [ -e "$dst" ]; then
        continue
    fi
    if [ -L "$src" ]; then
        cp -P "$src" "$dst"
    elif [ -f "$src" ]; then
        cp "$src" "$dst"
    else
        echo "  WARNING: $lib not found in SDK" >&2
    fi
    copied=$((copied + 1))
done
if [ $copied -gt 0 ]; then
    echo "  Copied $copied libraries ($(du -sh "$LIB_DIR" | cut -f1) total)"
else
    echo "  Libraries already present ($(du -sh "$LIB_DIR" | cut -f1))"
fi
echo

# --- Step 7: Download model + extract .const files ---
echo "Step 7: Downloading and preparing model ($MODEL)..."
python "$PROJECT_DIR/download_model.py" --model "$MODEL"
echo

# --- Final summary ---
NPU_DEVICE="${NPU_DEVICE:-/dev/accel/accel0}"
echo "=== Setup complete ==="
echo
echo "  Project:  $PROJECT_DIR"
echo "  Venv:     $VENV_DIR"
echo "  Libs:     $LIB_DIR"
echo "  Model:    $PROJECT_DIR/models/"
echo
if [ -e "$NPU_DEVICE" ]; then
    echo "  NPU:      $NPU_DEVICE (ready)"
else
    echo "  NPU:      $NPU_DEVICE NOT FOUND — see README.md for binding instructions."
fi
echo
echo "Generate an image:"
echo "  ./run.sh --model $MODEL \"a corgi sitting in a field of wildflowers\" --seed 42"
