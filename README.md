# Old Photo Restoration Studio v2

## Features

✅ **One-Click Restore** — "Restore Old Photo" button with auto-optimized presets
✅ **Preset Sliders** — Adjust color, face, and detail enhancement with intuitive sliders
✅ **Batch Mode** — Process entire folders of old photos at once
✅ **Remote API Support** — Send heavy processing to replicate.com or custom API servers
✅ **Local Fallback** — Works offline with Pillow-based enhancement
✅ **Dual Processing** — Local preprocessing + remote GPU inference

---

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Optional: Install local AI models (for offline 16GB VRAM processing)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install realesrgan gfpgan facexlib
```

---

## Run

```bash
python app.py
```

---

## Workflow: Local + Remote Hybrid

### Local Processing (No GPU needed)
- Load image → GUI applies color/contrast tuning locally
- Resize operations with Pillow
- Preview instantly

### Remote Processing (GPU-intensive, optional)
- If you provide an API URL (e.g., Replicate, custom server), the app sends the image for:
  - Real-ESRGAN upscaling
  - GFPGAN face restoration
  - Returns enhanced image to GUI

### How to use Remote API

1. **Get an API URL:**
   - **Replicate:** https://replicate.com/nightmareai/real-esrgan (free credits)
   - **Custom server:** Your own upscaling service

2. **In the GUI:**
   - Load image
   - Paste API URL in "Remote Processing → API URL" field
   - Check "Use remote API"
   - Click "Run Enhancement" or "Restore Old Photo"

3. **The app:**
   - Encodes image to base64
   - Sends to remote API
   - Receives upscaled/restored result
   - Shows preview + saves output

---

## Tabs

### Tab 1: Single Image
- Load an old photo
- **🎯 Restore Old Photo** — Auto preset (4x, faded colors, light face restore)
- **Color Enhancement** slider (0.5x - 2.5x)
- **Face Enhancement** slider (Off, Light, Strong)
- **Detail Enhancement** slider (0.5x - 2.5x)
- Optional remote API
- Before/after preview
- Save output

### Tab 2: Batch Restore
- Select folder with multiple old photos
- Choose output folder (or auto-create "restored" subfolder)
- Set scale (2x, 4x), denoise, workflow
- Click "Start Batch Restore"
- Progress bar + log of all processed images

---

## Preset Examples

### Faded Color Photo (Quick Restore)
```
Scale: 4x
Color: 1.35x (boosted for faded colors)
Face: Light (50%)
Detail: 1.30x
```

### Portrait (Quick Restore)
```
Scale: 4x
Color: 1.15x
Face: Strong (100%)
Detail: 1.25x
```

### Damaged/Low Quality (Advanced)
```
Scale: 2x (start small)
Color: 1.25x
Face: Off
Detail: 1.35x (high sharpness)
Denoise: ON
```

---

## Local AI Model Setup (16GB VRAM)

If you don't want to use a remote API:

1. Download models:
   ```bash
   # Real-ESRGAN (upscaler)
   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth
   mkdir models && mv RealESRGAN_x2plus.pth models/
   
   # GFPGAN (face restorer)
   wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.5/GFPGANv1.4.pth
   mv GFPGANv1.4.pth models/
   ```

2. The app will automatically detect and use them.

---

## FAQ

**Q: Can I use the app without installing heavy AI packages?**
A: Yes! The app works with just Pillow + PyQt5. Remote API optional. Local AI packages optional.

**Q: How long does batch processing take?**
A: Depends on image size and your system. ~2-5 seconds per image (local), ~10-30s per image (remote API).

**Q: Can I cancel batch processing?**
A: Not yet. Stop the GUI window to stop the batch.

**Q: Does remote API cost money?**
A: Depends on the service. Replicate offers free credits; custom APIs vary.

**Q: What formats are supported?**
A: PNG, JPG, JPEG, BMP, TIFF

---

## License

MIT
