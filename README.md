# Old Photo Restoration Studio

A desktop app for restoring and enhancing old colored photos using a practical AI-assisted workflow.

## What this project includes

- Qt-based desktop GUI
- load single image and preview before/after
- optional Real-ESRGAN/GFPGAN integration
- fallback enhancement pipeline using Pillow when AI packages are not installed
- presets for faded color, portrait, and damaged photos
- save final image as PNG/JPG/TIFF

## Quick start

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Optional AI packages

For the best output on a 16GB VRAM system, install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install realesrgan gfpgan facexlib
```

## Recommended workflow for old color photos

1. Denoise
2. Correct faded colors
3. Run 2x or 4x upscale
4. Restore faces if needed
5. Final sharpen and export

## Notes

This app is designed to work immediately out of the box with a standard Python environment, while still supporting more advanced external AI models when available.
