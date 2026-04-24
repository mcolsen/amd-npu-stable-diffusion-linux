# amd-npu-stable-diffusion-linux

Stable Diffusion image generation on AMD Ryzen AI XDNA 2 NPUs for Linux.

Loads AMD's pre-compiled ONNX models from the Ryzen AI SDK and runs the 
diffusion pipeline (text encoder, UNet, VAE decoder) with ~98% of UNet
operations offloaded to the NPU.

## Supported models

| Preset      | Model              | Steps | Time  | Resolution | Notes                                                      |
|-------------|--------------------|-------|-------|------------|-----------------------------------------------------------|
| `sd15`      | Stable Diffusion 1.5 | 20    | ~5 s  | 512×512    | Classifier-free guidance (batch = 2 UNet)                  |
| `sd-turbo`  | SD-Turbo           | 1     | ~2 s  | 512×512    | Single-step distilled, no guidance                         |
| `sdxl-base` | SDXL-Base          | 30    | ~35 s | 1024×1024  | Two text encoders + micro-conditioning; CFG mandatory      |
| `sdxl-turbo`| SDXL-Turbo         | 1     | ~2 s  | 512×512    | Two text encoders, single-step, no CFG                     |
| `vega`      | Segmind Vega       | 20    | ~12 s | 1024×1024  | Distilled SDXL, two text encoders, CFG mandatory           |

Times measured on a Ryzen AI MAX+ 395 (Strix Halo) with 128 GB RAM, Fedora 43,
kernel 6.19.

## Hardware & OS

Tested on:

- AMD Ryzen AI MAX+ 395 (Strix Halo) / 128 GB RAM
- Fedora 43, kernel 6.19
- amdxdna DKMS module, firmware 1.1.2.65

Should also work on other Ryzen AI 300-series chips (Strix Point etc.). You'll
need a working `/dev/accel/accel0` NPU device and the Ryzen AI SDK's ONNX
Runtime. If you bring it up on another SKU, please open an issue / PR with the
results.

## One-time prerequisites

Do these once per machine. None of them are managed by `setup.sh`.

### 1. Install the amdxdna driver + XRT

Build and install from [amd/xdna-driver](https://github.com/amd/xdna-driver).
Follow their `dkms_driver.sh` flow. The in-tree module shipped with recent
kernels aborts DD xclbins.

### 2. Bind the NPU device

```bash
# Check if /dev/accel/accel0 already exists
ls /dev/accel/accel0

# If not, discover the NPU's PCI address
lspci -d 1022: | grep -i 'signal\|npu'
# e.g. "c4:00.1 Signal processing controller: AMD NPU ..."

# Bind it (substitute the BDF you just found; domain 0000 is fine for a single-NPU box)
echo "0000:c4:00.1" | sudo tee /sys/bus/pci/drivers/amdxdna/bind
```

### 3. Raise the NPU command-watchdog timeout to 30 s

The `amdxdna` driver defaults to a **2-second** per-command watchdog. SDXL
(~1 s/step in a single fused 1742-op kernel) reliably exceeds it; SD1.5
intermittently overshoots on cold-start too. Make the fix persistent:

```bash
echo 'options amdxdna timeout_in_sec=30' | sudo tee /etc/modprobe.d/amdxdna.conf
sudo rmmod amdxdna && sudo modprobe amdxdna
cat /sys/module/amdxdna/parameters/timeout_in_sec   # should print 30
```

**Important:** the firmware latches the timeout at driver init, so writing to
`/sys/module/amdxdna/parameters/timeout_in_sec` at runtime does *not* take
effect. The sysfs value can read `30` while the firmware is still on 2 s. If
you hit `ERT_CMD_STATE_TIMEOUT` on any model (even SD1.5), reload the module
regardless of what sysfs says.

### 4. Install the Ryzen AI SDK 1.7.1

Download from [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/) and extract
anywhere on the system. `setup.sh` auto-detects `~/ryzen_ai-1.7.1` and
`/opt/ryzen_ai*`; for anything else set `RYZEN_AI_PATH=/path/to/sdk` before
running setup.

You'll also need whichever Python the SDK's bundled ONNX Runtime was built for
(Python 3.12 for 1.7.1). `setup.sh` will tell you if it's missing.

### 5. Request Hugging Face access for the models you want

The models are pre-compiled and redistributed by AMD. Four of the five presets
are gated (access must be requested on the model page); Segmind Vega is public.

| Preset       | Hugging Face repo                                                            | Access |
|--------------|-------------------------------------------------------------------------------|--------|
| `sd15`       | [amd/stable-diffusion-1.5-amdnpu](https://huggingface.co/amd/stable-diffusion-1.5-amdnpu) | Gated  |
| `sd-turbo`   | [amd/sd-turbo-amdnpu](https://huggingface.co/amd/sd-turbo-amdnpu)             | Gated  |
| `sdxl-base`  | [amd/sdxl-base-amdnpu](https://huggingface.co/amd/sdxl-base-amdnpu)           | Gated  |
| `sdxl-turbo` | [amd/sdxl-turbo-amdnpu](https://huggingface.co/amd/sdxl-turbo-amdnpu)         | Gated  |
| `vega`       | [amd/segmind-vega-amdnpu](https://huggingface.co/amd/segmind-vega-amdnpu)     | Public |

Click "Request access" on each gated model and authenticate locally:

```bash
hf auth login
```

## Quick start

```bash
# Install everything and download the default SD1.5 model (takes ~10 min; model is ~950 MB)
./setup.sh

# Generate an image
./run.sh "a corgi sitting in a field of wildflowers" --seed 42
```

## Using other models

Each preset's model is downloaded separately. You can either pick one at setup
time, or run the downloader afterwards for additional models.

```bash
# Install + download Segmind Vega in one shot
./setup.sh --model vega

# Or, after initial setup, add more models whenever
python download_model.py --model sdxl-turbo
python download_model.py --model sdxl-base

# Then generate
./run.sh --model vega "a corgi in a field of wildflowers" --seed 42
./run.sh --model sdxl-turbo "a cat on a windowsill" --seed 42
./run.sh --model sdxl-base "an astronaut riding a horse" --seed 42
```

## Options

```
./run.sh [--model PRESET] PROMPT [flags...]

  --model {sd15,sd-turbo,sdxl-base,sdxl-turbo,vega}   Model preset (default: sd15)
  --negative-prompt TEXT                              Negative prompt
  --steps N                                           Denoising steps (default: per model)
  --guidance-scale F                                  CFG scale (default: per model)
  --seed N                                            Random seed
  --output PATH                                       Output image (default: output.png)
  --model-dir PATH                                    Override resolved model directory
```

### Environment variables

| Variable           | Purpose                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------|
| `RYZEN_AI_PATH`    | Ryzen AI SDK install directory (used by `setup.sh` if not in the default locations)                 |
| `NPUTOP_PATH`      | Local nputop checkout to install editable instead of pulling from git                               |
| `NPU_DEVICE`       | NPU device node (default `/dev/accel/accel0`)                                                       |
| `XDNA_MODELS_DIR`  | Parent directory for downloaded models (default `./models/`)                                        |
| `XDNA_CACHE_DIR`   | Root for per-run telemetry CSVs (default `~/.cache/xdna2-npu-diffusion`)                            |

## Performance

Measured on AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB RAM, Fedora 43:

### SD1.5 (20 steps, guidance = 7.5)

| Stage                          | Time            |
|--------------------------------|-----------------|
| Model loading                  | 1.0 s           |
| Text encoding (CPU)            | 0.1 s           |
| UNet denoising (20 steps, NPU) | 3.6 s (5.5 it/s)|
| VAE decode (NPU)               | 0.2 s           |
| **Total**                      | **~5 s**        |

### SD-Turbo (1 step, no guidance)

| Stage                          | Time   |
|--------------------------------|--------|
| Model loading                  | 1.3 s  |
| Text encoding (CPU)            | 0.2 s  |
| UNet denoising (1 step, NPU)   | 0.1 s  |
| VAE decode (NPU)               | 0.2 s  |
| **Total**                      | **~2 s**|

## Troubleshooting

**Black / all-zero image.** Something wrote `compile_fusion_rt=1` to the
session options at some point and overwrote the AMD metastate files with
broken versions. Delete the affected `models/<preset>/` directory and run
`python download_model.py --model <preset>` again. Never add
`compile_fusion_rt` to the session config.

**`ERT_CMD_STATE_TIMEOUT` from the driver.** The firmware watchdog fired.
Confirm the driver was reloaded after setting `timeout_in_sec=30`
(prerequisite step 3). Sysfs can read `30` while the firmware is still on 2 s;
only a `rmmod` + `modprobe` takes effect.

**`unordered_map::at` during UNet step 1.** The `amdxdna` module loaded but
the NPU wasn't actually bound. Re-run the `lspci` + `bind` dance in
prerequisite step 2.

**`401 Unauthorized` from Hugging Face.** You haven't been granted access to
the gated model, or haven't run `hf auth login`. Visit the model
page, click "Request access", wait for approval, then authenticate.

**"ONNX Runtime not found" during setup.** The SDK path doesn't contain
`lib/python3.X/site-packages/onnxruntime`. Confirm `RYZEN_AI_PATH` points at a
full SDK install (not just the `deployment/` subdirectory).

**No NPU telemetry in diagnostics output.** `nputop` either isn't installed
or can't open `/dev/accel/accel0`. Check that your user has read/write access
to the device node. Telemetry is optional; the pipeline still works without
it.

**Where are the telemetry CSVs?**
`~/.cache/xdna2-npu-diffusion/telemetry/<run_id>.csv` by default.
Override with `XDNA_CACHE_DIR`.

## Project structure

```
xdna2-npu-diffusion/
├── setup.sh                # One-time bootstrap (venv, libs, model, const extraction)
├── run.sh                  # Wrapper that sets env and runs the pipeline
├── run_npu.py              # Unified text-to-image pipeline (all 5 presets)
├── diagnostics.py          # NPU health probes, telemetry CSV, retry wrapper
├── download_model.py       # Pull a preset from Hugging Face and extract .const files
├── setup_const_files.py    # .fconst → .const byte-slice extractor
├── requirements.txt        # Runtime Python deps (excluding ORT and PyTorch-CPU)
├── tests/
│   └── test_dd_smoke.py    # DD custom ops smoke test on random UNet inputs
├── lib/                    # [generated by setup.sh] Deployment libraries from the SDK
└── models/                 # [generated by setup.sh] Downloaded pre-compiled ONNX models
```

## Important notes

- **Never set `compile_fusion_rt=1`** as a session config option. It overwrites
  AMD's pre-compiled metastate files with broken versions that produce black
  images.
- The NPU device must be bound before running. If the `amdxdna` module loads
  but doesn't bind the device, all DD operations fail with a cryptic
  `unordered_map::at` error.

## Acknowledgments

- **AMD** for publishing the pre-compiled DynamicDispatch ONNX models on
  Hugging Face and the Ryzen AI SDK runtime, as well as for the [Windows-only
  implementation](https://github.com/amd/sd-sandbox) that got me wondering
  if I could get this working on Linux. 
