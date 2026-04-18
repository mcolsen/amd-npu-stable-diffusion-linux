#!/usr/bin/env python3
"""
Stable Diffusion text-to-image on AMD Strix Halo NPU.

Uses AMD's pre-compiled ONNX models with DynamicDispatch custom ops
to run diffusion pipelines on the NPU. Supports multiple models via
the --model flag.

Usage (via wrapper):
  ./run.sh "a corgi sitting in a field of wildflowers" --seed 42
  ./run.sh --model sd-turbo "a cat on a windowsill" --seed 42
  ./run.sh --model vega "a corgi in a field of wildflowers" --seed 42

Usage (manual):
  source .venv/bin/activate
  export DD_ROOT=./lib
  export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
  python run_npu.py "a corgi sitting in a field of wildflowers"
  python run_npu.py --model sd-turbo "a cat on a windowsill"
  python run_npu.py --model vega "a corgi in a field of wildflowers"
"""

import argparse
import ctypes
import importlib
import json
import os
import sys
import time

import numpy as np
import onnxruntime
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import diagnostics

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CUSTOM_OP_LIB = os.path.join(_SCRIPT_DIR, "lib", "libonnx_custom_ops.so")

# --- Model presets ---
MODELS = {
    "sd15": {
        "name": "Stable Diffusion 1.5",
        "dir_name": "stable-diffusion-1.5-amdnpu",
        "default_steps": 20,
        "default_guidance_scale": 7.5,
        "family": "sd15",
        "latent_size": 64,
        "force_cfg": False,
    },
    "sd-turbo": {
        "name": "SD-Turbo",
        "dir_name": "sd-turbo-amdnpu",
        "default_steps": 1,
        "default_guidance_scale": 0.0,
        "family": "sd15",
        "latent_size": 64,
        "force_cfg": False,
    },
    "sdxl-base": {
        "name": "SDXL-Base",
        "dir_name": "sdxl-base-amdnpu",
        "default_steps": 30,
        "default_guidance_scale": 5.0,
        "family": "sdxl",
        "latent_size": 128,
        # SDXL-Base's compiled UNet has batch=2 baked in; CFG is mandatory.
        "force_cfg": True,
    },
    "sdxl-turbo": {
        "name": "SDXL-Turbo",
        "dir_name": "sdxl-turbo-amdnpu",
        "default_steps": 1,
        "default_guidance_scale": 0.0,
        "family": "sdxl",
        "latent_size": 64,
        # SDXL-Turbo's compiled UNet has batch=1 baked in; CFG is not supported.
        "force_cfg": False,
    },
    "vega": {
        "name": "Segmind Vega",
        "dir_name": "segmind-vega-amdnpu",
        "default_steps": 20,
        "default_guidance_scale": 7.5,
        "family": "sdxl",
        "latent_size": 128,
        # Distilled SDXL; compiled UNet is batch=2 like SDXL-Base — CFG mandatory.
        "force_cfg": True,
    },
}

# SDXL micro-conditioning row: [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
# Values aren't baked into the compiled graph (only the [*,6] shape is), so we
# derive them from the model's native output resolution (latent_size * 8).


def load_onnx_model(model_dir, subfolder, model_file, custom_op_path):
    """Load an ONNX model with DD custom ops registered."""
    dd_path = os.path.join(model_dir, subfolder, "dd")
    onnx_path = os.path.join(dd_path, model_file)

    os.makedirs(os.path.join(dd_path, ".cache"), exist_ok=True)

    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_session_config_entry("dd_cache", os.path.join(dd_path, ".cache"))
    so.add_session_config_entry("onnx_custom_ops_const_key", onnx_path)
    so.register_custom_ops_library(custom_op_path)

    return onnxruntime.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def encode_prompts_sd15(tokenizer, text_encoder, prompt, negative_prompt, use_cfg):
    """Encode with a single CLIP text encoder (SD1.5 / SD-Turbo)."""
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]

    if use_cfg:
        uncond_input = tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return {"encoder_hidden_states": text_embeddings}


def _sdxl_encode_one(tokenizers, text_encoders, text):
    """Run one text through both SDXL encoders. Returns (hidden [1,77,2048], pooled [1,1280])."""
    hidden_parts = []
    pooled = None
    for i, (tok, enc) in enumerate(zip(tokenizers, text_encoders)):
        ids = tok(
            [text],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        with torch.no_grad():
            out = enc(ids, output_hidden_states=True)
        # SDXL convention: penultimate hidden state from both encoders.
        hidden_parts.append(out.hidden_states[-2])
        # Pooled embed comes from text_encoder_2 (CLIPTextModelWithProjection).
        if i == 1:
            pooled = out.text_embeds
    return torch.cat(hidden_parts, dim=-1), pooled


def encode_prompts_sdxl(tokenizers, text_encoders, prompt, negative_prompt, use_cfg, target_size):
    """Encode with both SDXL text encoders. Produces batch=2 under CFG, batch=1 otherwise."""
    cond_hidden, cond_pooled = _sdxl_encode_one(tokenizers, text_encoders, prompt)
    time_ids_row = (target_size, target_size, 0, 0, target_size, target_size)

    if use_cfg:
        uncond_hidden, uncond_pooled = _sdxl_encode_one(tokenizers, text_encoders, negative_prompt)
        encoder_hidden_states = torch.cat([uncond_hidden, cond_hidden]).to(torch.float32)
        text_embeds = torch.cat([uncond_pooled, cond_pooled]).to(torch.float32)
        time_ids = torch.tensor([time_ids_row, time_ids_row], dtype=torch.int64)
    else:
        encoder_hidden_states = cond_hidden.to(torch.float32)
        text_embeds = cond_pooled.to(torch.float32)
        time_ids = torch.tensor([time_ids_row], dtype=torch.int64)

    return {
        "encoder_hidden_states": encoder_hidden_states,
        "text_embeds": text_embeds,
        "time_ids": time_ids,
    }


_ORT_TO_NP_DTYPE = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def session_input_dtypes(session):
    """Map input name → numpy dtype from the ONNX session's declared types.

    Different compiled UNets disagree on per-input dtypes even within a family
    (e.g., SDXL-Base wants timestep=float32 / time_ids=int64, Segmind Vega wants
    timestep=int64 / time_ids=float32). Driving casts from the session itself
    means new model variants just work.
    """
    return {i.name: _ORT_TO_NP_DTYPE[i.type] for i in session.get_inputs()}


def build_unet_inputs(family, latent_model_input, t, prompt_tensors, dtypes):
    """Build the ONNX input dict for one UNet step, branching on model family."""
    inputs = {
        "sample": latent_model_input.numpy().astype(dtypes["sample"], copy=False),
        "timestep": np.array([t.item()], dtype=dtypes["timestep"]),
        "encoder_hidden_states": prompt_tensors["encoder_hidden_states"]
            .numpy().astype(dtypes["encoder_hidden_states"], copy=False),
    }
    if family == "sdxl":
        inputs["text_embeds"] = prompt_tensors["text_embeds"].numpy().astype(
            dtypes["text_embeds"], copy=False)
        inputs["time_ids"] = prompt_tensors["time_ids"].numpy().astype(
            dtypes["time_ids"], copy=False)
    return inputs


def resolve_model_dir(model_name, override_dir):
    """Find the model directory.

    Resolution order: --model-dir override > $XDNA_MODELS_DIR/<preset> > ./models/<preset>.
    """
    if override_dir:
        return override_dir
    info = MODELS[model_name]
    env_root = os.environ.get("XDNA_MODELS_DIR")
    if env_root:
        return os.path.join(env_root, info["dir_name"])
    return os.path.join(_SCRIPT_DIR, "models", info["dir_name"])


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion on AMD NPU")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument(
        "--model", type=str, choices=MODELS.keys(), default="sd15",
        help="Model preset (default: sd15)",
    )
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=None, help="Denoising steps (default: per model)")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale (default: per model)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--model-dir", type=str, default=None, help="Override model directory")
    parser.add_argument("--custom-op-path", type=str, default=_DEFAULT_CUSTOM_OP_LIB)
    args = parser.parse_args()

    # Resolve per-model defaults
    model_info = MODELS[args.model]
    family = model_info["family"]
    steps = args.steps if args.steps is not None else model_info["default_steps"]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else model_info["default_guidance_scale"]
    model_dir = resolve_model_dir(args.model, args.model_dir)
    use_cfg = model_info["force_cfg"] or guidance_scale > 1.0
    if model_info["force_cfg"] and guidance_scale <= 1.0:
        print(f"Note: {model_info['name']} has CFG baked into the compiled graph; "
              f"guidance_scale={guidance_scale} will still run with a 2x batch.")
    # SDXL-Turbo's compiled UNet is batch=1 — CFG would need a batch=2 input and
    # would fail deep inside ORT with an opaque shape mismatch. Reject up front.
    if family == "sdxl" and not model_info["force_cfg"] and use_cfg:
        print(f"Error: {model_info['name']} has a batch=1 compiled UNet and does not "
              f"support classifier-free guidance. Use --guidance-scale 0.0.")
        sys.exit(1)

    # Validate
    if not os.path.exists(args.custom_op_path):
        print(f"Error: custom ops library not found: {args.custom_op_path}")
        print("  Run ./setup.sh to stage it from the Ryzen AI SDK.")
        sys.exit(1)
    npu_device = os.environ.get("NPU_DEVICE", "/dev/accel/accel0")
    if not os.path.exists(npu_device):
        print(f"Error: NPU device {npu_device} not found.")
        print("  Find the NPU's PCI address:  lspci -d 1022: | grep -i 'signal\\|npu'")
        print("  Bind it (replace BDF):       echo <BDF> | sudo tee /sys/bus/pci/drivers/amdxdna/bind")
        sys.exit(1)
    if not os.path.exists(model_dir):
        print(f"Error: model directory not found: {model_dir}")
        print(f"  Run: python download_model.py --model {args.model}")
        sys.exit(1)

    t_start = time.perf_counter()
    print(f"Model: {model_info['name']} ({steps} steps, guidance={guidance_scale})")

    # Preflight: wake NPU from PM suspend, check clock, surface iGPU contention.
    # The NpuDevice handle is held for the whole run so autosuspend can't drop
    # it mid-inference; it's also the source for per-step telemetry.
    npu = diagnostics.wake_npu()
    diagnostics.preflight(npu)

    # Load custom ops library
    ctypes.CDLL(args.custom_op_path)

    # --- Text Encoders (CPU) ---
    # SDXL needs two: CLIPTextModel + CLIPTextModelWithProjection (for pooled output).
    print("Loading text encoder(s)...")
    tokenizers = [CLIPTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))]
    text_encoders = [CLIPTextModel.from_pretrained(os.path.join(model_dir, "text_encoder"))]
    if family == "sdxl":
        tokenizers.append(
            CLIPTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer_2"))
        )
        text_encoders.append(
            CLIPTextModelWithProjection.from_pretrained(os.path.join(model_dir, "text_encoder_2"))
        )

    # --- UNet (NPU) ---
    print("Loading UNet (NPU)...")
    unet = load_onnx_model(model_dir, "unet", "replaced.onnx", args.custom_op_path)
    unet_dtypes = session_input_dtypes(unet)

    # --- VAE Decoder (NPU) ---
    print("Loading VAE decoder (NPU)...")
    vae = load_onnx_model(model_dir, "vae_decoder", "replaced.onnx", args.custom_op_path)

    # --- Scheduler ---
    scheduler_config = json.load(
        open(os.path.join(model_dir, "scheduler", "scheduler_config.json"))
    )
    scheduler_cls = getattr(
        importlib.import_module("diffusers.schedulers"),
        scheduler_config["_class_name"],
    )
    scheduler = scheduler_cls.from_pretrained(os.path.join(model_dir, "scheduler"))

    # VAE scaling factor
    vae_config = json.load(open(os.path.join(model_dir, "vae_decoder", "config.json")))
    vae_scaling_factor = vae_config.get("scaling_factor", 0.18215)

    t_load = time.perf_counter() - t_start
    print(f"Models loaded in {t_load:.2f}s")

    # --- Encode prompt ---
    print("Encoding prompt...")
    if family == "sdxl":
        prompt_tensors = encode_prompts_sdxl(
            tokenizers, text_encoders, args.prompt, args.negative_prompt,
            use_cfg, target_size=model_info["latent_size"] * 8,
        )
    else:
        prompt_tensors = encode_prompts_sd15(
            tokenizers[0], text_encoders[0], args.prompt, args.negative_prompt, use_cfg
        )

    # --- Prepare latents ---
    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    latent_size = model_info["latent_size"]
    latents = torch.randn((1, 4, latent_size, latent_size), generator=generator)
    scheduler.set_timesteps(steps)
    latents = latents * scheduler.init_noise_sigma

    # --- Denoise loop ---
    print(f"Denoising ({steps} steps)...")
    telemetry = diagnostics.TelemetryWriter(npu=npu)
    print(f"  Telemetry: {telemetry.path}")
    t_denoise = time.perf_counter()
    try:
        for i, t in enumerate(scheduler.timesteps):
            if use_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            step_t0 = time.perf_counter()
            noise_pred = diagnostics.run_with_retry(
                unet, None,
                build_unet_inputs(family, latent_model_input, t, prompt_tensors, unet_dtypes),
                label=f"UNet step {i+1}/{steps}", npu=npu,
            )[0]
            telemetry.record("unet", i + 1, time.perf_counter() - step_t0)

            noise_pred = torch.from_numpy(noise_pred)

            if use_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = scheduler.step(noise_pred, t, latents).prev_sample

            print(f"  Step {i+1}/{steps}", end="\r")

        t_denoise = time.perf_counter() - t_denoise
        print(f"  Denoising done in {t_denoise:.2f}s ({steps/t_denoise:.1f} it/s)")

        # --- VAE decode ---
        print("Decoding latents...")
        latents_scaled = (1 / vae_scaling_factor) * latents
        # SD1.5/Turbo VAE input is named `latents`; SDXL's is `latent_sample`.
        vae_input_name = "latent_sample" if family == "sdxl" else "latents"
        t_vae_start = time.perf_counter()
        image_array = diagnostics.run_with_retry(
            vae, None, {vae_input_name: latents_scaled.numpy()},
            label="VAE decode", npu=npu,
        )[0]
        t_vae = time.perf_counter() - t_vae_start
        telemetry.record("vae", 1, t_vae)
        print(f"  VAE decode: {t_vae:.2f}s")
    finally:
        telemetry.close()
        if npu is not None:
            npu.close()

    # --- Save image ---
    image_array = np.clip((image_array + 1) / 2, 0, 1)
    image_array = (image_array[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(args.output)

    t_total = time.perf_counter() - t_start
    print(f"\nSaved: {args.output}")
    print(f"Total: {t_total:.2f}s (load: {t_load:.2f}s, denoise: {t_denoise:.2f}s, vae: {t_vae:.2f}s)")


if __name__ == "__main__":
    main()
