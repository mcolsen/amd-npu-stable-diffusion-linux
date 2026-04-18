#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run Stable Diffusion on the NPU with the correct environment.
#
# Usage:
#   ./run.sh "a corgi sitting in a field of wildflowers" --seed 42
#   ./run.sh --model sd-turbo "a cat on a windowsill" --seed 42
#   ./run.sh --model sdxl-turbo "a cat on a windowsill" --seed 42
#   ./run.sh "an astronaut riding a horse" --steps 30 --output astronaut.png
#
# Override NPU device location with NPU_DEVICE=/dev/accel/accel1 ./run.sh ...

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
LIB_DIR="$PROJECT_DIR/lib"
NPU_DEVICE="${NPU_DEVICE:-/dev/accel/accel0}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: venv not found at $VENV_DIR. Run ./setup.sh first." >&2
    exit 1
fi

if [ ! -f "$LIB_DIR/libonnx_custom_ops.so" ]; then
    echo "Error: deployment libraries not found in $LIB_DIR. Run ./setup.sh first." >&2
    exit 1
fi

if [ ! -e "$NPU_DEVICE" ]; then
    echo "Error: NPU device $NPU_DEVICE not found." >&2
    echo "  The amdxdna driver must be loaded and the NPU bound." >&2
    echo "  1. Find the NPU's PCI address:" >&2
    echo "       lspci -d 1022: | grep -i 'signal\\|npu'" >&2
    echo "  2. Bind it (replace BDF with the address from step 1):" >&2
    echo "       echo <BDF> | sudo tee /sys/bus/pci/drivers/amdxdna/bind" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export DD_ROOT="$LIB_DIR"
export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec python "$PROJECT_DIR/run_npu.py" "$@"
