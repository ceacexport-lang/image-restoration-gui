from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pick_model_path(candidates):
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return str(p)
    return None


def _build_model_candidates(name: str):
    home = Path.home()
    roots = [
        Path.cwd() / "models",
        Path.cwd() / "weights",
        Path.cwd(),
        home / ".cache",
        home / ".models",
    ]
    results = []
    for root in roots:
        results.append(str(root / name))
        results.append(str(root / "realesrgan" / name))
        results.append(str(root / "gfpgan" / name))
    return results


def _try_real_esrgan(image_path: str, output_path: str, scale: int = 2, use_gpu: bool = True):
    if not _has_module("realesrgan"):
        return False, "Real-ESRGAN package not installed."

    model_name = "realesrgan-x2plus.pth" if scale == 2 else "realesrgan-x4plus.pth"
    model_path = _pick_model_path(_build_model_candidates(model_name))
    if not model_path:
        return False, "Real-ESRGAN model file not found. Download it into ./models or ./weights."

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    except Exception as exc:
        return False, f"Real-ESRGAN import failed: {exc}"

    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        model = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=None,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=0 if use_gpu else None,
        )
        output, _ = model.enhance(img_np, outscale=scale)
        result = Image.fromarray(output)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_file)
        return True, str(output_file)
    except Exception as exc:
        return False, f"Real-ESRGAN execution failed: {exc}"


def _try_gfpgan(image_path: str, output_path: str):
    if not _has_module("gfpgan"):
        return False, "GFPGAN package not installed."

    candidates = _build_model_candidates("GFPGANv1.4.pth")
    model_path = _pick_model_path(candidates)
    if not model_path:
        return False, "GFPGAN model file not found. Download it to ./models or ./weights."

    try:
        from gfpgan import GFPGANer
    except Exception as exc:
        return False, f"GFPGAN import failed: {exc}"

    try:
        img = Image.open(image_path).convert("RGB")
        restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        output, _ = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        result = Image.fromarray(output)
        result.save(output_path)
        return True, output_path
    except Exception as exc:
        return False, f"GFPGAN execution failed: {exc}"


def _fallback_enhancement(input_path: str, output_path: str, scale: int = 2, denoise: bool = True,
                          color_boost: float = 1.2, contrast: float = 1.1, sharpness: float = 1.2,
                          face_restore: bool = False, workflow: str = "Balanced Restore") -> str:
    image = Image.open(input_path).convert("RGB")

    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    if workflow in {"Faded Color Restore", "Modern Fresh Look", "Portrait Refresh"}:
        image = image.filter(ImageFilter.SHARPEN)

    if scale >= 4:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

    image = ImageEnhance.Color(image).enhance(color_boost)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    if face_restore:
        image = image.filter(ImageFilter.SHARPEN)
        image = image.resize((image.width + 20, image.height + 20), Image.Resampling.LANCZOS)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return str(output)


def enhance_image(input_path: str, output_path: str, scale: int = 2, denoise: bool = True,
                  color_boost: float = 1.2, contrast: float = 1.1, sharpness: float = 1.2,
                  face_restore: bool = False, workflow: str = "Balanced Restore") -> tuple[bool, str, str]:
    """Try Real-ESRGAN first, then GFPGAN, then fall back to Pillow-based enhancement.
    Returns: (success, output_path, message)
    """
    try:
        if scale in {2, 4}:
            used, message = _try_real_esrgan(input_path, output_path, scale=scale, use_gpu=True)
            if used:
                return True, output_path, "Real-ESRGAN applied"
            if face_restore:
                used, msg2 = _try_gfpgan(output_path, output_path)
                if used:
                    return True, output_path, msg2
            fallback_path = _fallback_enhancement(
                input_path,
                output_path,
                scale=scale,
                denoise=denoise,
                color_boost=color_boost,
                contrast=contrast,
                sharpness=sharpness,
                face_restore=face_restore,
                workflow=workflow,
            )
            return True, fallback_path, f"Fallback pipeline used: {message}"

        fallback_path = _fallback_enhancement(
            input_path,
            output_path,
            scale=scale,
            denoise=denoise,
            color_boost=color_boost,
            contrast=contrast,
            sharpness=sharpness,
            face_restore=face_restore,
            workflow=workflow,
        )
        return True, fallback_path, "Fallback pipeline used"
    except Exception as exc:
        fallback_path = _fallback_enhancement(
            input_path,
            output_path,
            scale=scale,
            denoise=denoise,
            color_boost=color_boost,
            contrast=contrast,
            sharpness=sharpness,
            face_restore=face_restore,
            workflow=workflow,
        )
        return True, fallback_path, f"Recovered with fallback pipeline after error: {exc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhance old photos with optional AI restoration.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--face-restore", action="store_true")
    parser.add_argument("--workflow", default="Balanced Restore")
    args = parser.parse_args()

    ok, out, msg = enhance_image(
        args.input,
        args.output,
        scale=args.scale,
        denoise=args.denoise,
        color_boost=1.2,
        contrast=1.1,
        sharpness=1.2,
        face_restore=args.face_restore,
        workflow=args.workflow,
    )
    print(f"ok={ok}, output={out}, message={msg}")







































































































































































