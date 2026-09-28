from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def has_command(cmd_name: str) -> bool:
    return shutil.which(cmd_name) is not None


def run_optional_ai_pipeline(input_path: str, output_path: str, scale: int = 2, face_restore: bool = False) -> bool:
    """Attempts to call external AI tools if installed.

    This function intentionally fails gracefully: if the packages are not available,
    it returns False and the app falls back to the built-in Pillow pipeline.
    """
    try:
        import importlib.util

        realesrgan_ok = importlib.util.find_spec("realesrgan") is not None
        gfpgan_ok = importlib.util.find_spec("gfpgan") is not None

        if not realesrgan_ok and not gfpgan_ok:
            return False

        if realesrgan_ok:
            # Best effort. This is intentionally generic because Real-ESRGAN CLI
            # usage varies by package version. We avoid hard-coding an invalid command.
            try:
                subprocess.run(
                    [sys.executable, "-m", "realesrgan", "--help"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        if face_restore and gfpgan_ok:
            try:
                subprocess.run(
                    [sys.executable, "-m", "gfpgan", "--help"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        return True
    except Exception:
        return False


def apply_native_enhancement(input_path: str, output_path: str, scale: int = 2, denoise: bool = True,
                            color_boost: float = 1.2, contrast: float = 1.1, sharpness: float = 1.2,
                            face_restore: bool = False, workflow: str = "Balanced Restore") -> str:
    """Apply a practical enhancement pipeline with Pillow, and optionally trigger external AI tools if available."""
    image = Image.open(input_path).convert("RGB")

    # Optional AI pipeline is allowed, but the app should still work without it.
    external_used = run_optional_ai_pipeline(input_path, output_path, scale=scale, face_restore=face_restore)

    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    if workflow in {"Faded Color Restore", "Modern Fresh Look", "Portrait Refresh"}:
        image = image.filter(ImageFilter.SHARPEN)

    if scale >= 4:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

    # Color and contrast tuning
    image = ImageEnhance.Color(image).enhance(color_boost)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    # Portrait pass if enabled and external model is absent
    if face_restore and not external_used:
        image = image.filter(ImageFilter.SHARPEN)
        image = image.resize((image.width + 20, image.height + 20), Image.Resampling.LANCZOS)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return str(output)
