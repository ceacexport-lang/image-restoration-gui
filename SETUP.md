# Setup & Run Guide

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- 16GB VRAM GPU (recommended for Real-ESRGAN, but CPU fallback works)

## Step 1: Clone the Repository

```bash
git clone https://github.com/ceacexport-lang/image-restoration-gui.git
cd image-restoration-gui
```

## Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: (Optional) Install AI Model Support

If you want to use **Real-ESRGAN** and **GFPGAN** locally, install additional packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install realesrgan gfpgan facexlib
```

**Note:** Replace `cu118` with `cu121` for CUDA 12.1, or `cpu` if you don't have a GPU.

## Step 5: Run the GUI

```bash
python app.py
```

The GUI window will open. You can now:
1. **Load Image** - Select an old photo
2. **Configure options** - Choose upscaling, denoise, color boost, etc.
3. **Run Enhancement** - Click "Run Enhancement" to process
4. **Save Output** - Export the restored image

---

## How to Use the GUI

### Basic Workflow

1. **Load an image:**
   - Click "Load Image" and select your old photo (PNG, JPG, BMP, TIFF)
   - Preview appears on the left side

2. **Choose enhancement settings:**
   - **Upscale:** 2x or 4x resolution increase
   - **Face restoration:** Enable for portraits (requires GFPGAN)
   - **Denoise:** Removes grain and noise
   - **Color Boost:** Enhance faded colors (1.0 = no change, 1.5 = strong boost)
   - **Contrast:** Make image punchier (1.0 = no change, 1.2 = recommended)
   - **Sharpness:** Add detail clarity (1.0 = no change, 1.2-1.5 = recommended)
   - **Workflow:** Select restoration style:
     - "Real-ESRGAN + color repair" - Best for faded photos
     - "Modern fresh look" - Makes old photos look contemporary
     - "Balanced restore" - Middle ground
     - "High detail restore" - Maximum detail extraction

3. **Run enhancement:**
   - Click "Run Enhancement"
   - Progress bar shows processing status
   - Before/after preview updates automatically
   - Takes 30 seconds to 2 minutes depending on image size

4. **Save the result:**
   - Click "Save Output"
   - Choose format (PNG, JPEG, TIFF)
   - Select save location

---

## Recommended Settings by Image Type

### Faded Color Photos
```
Upscale: 4x
Denoise: ✓
Color Boost: 1.2-1.3
Contrast: 1.15
Sharpness: 1.25
Workflow: Real-ESRGAN + color repair
```

### Old Portraits
```
Upscale: 4x
Denoise: ✓
Color Boost: 1.15
Contrast: 1.1
Sharpness: 1.3
Face restoration: ✓
Workflow: Modern fresh look
```

### Highly Damaged / Low Quality
```
Upscale: 2x (start low)
Denoise: ✓
Color Boost: 1.25
Contrast: 1.2
Sharpness: 1.5
Workflow: High detail restore
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'PyQt5'"
```bash
pip install PyQt5
```

### Image won't load
- Ensure file format is supported (PNG, JPG, BMP, TIFF)
- Check file isn't corrupted

### Enhancement is slow
- Reduce image size first
- Use 2x instead of 4x upscaling
- Disable face restoration if not needed
- Ensure GPU is available if installed

### Want faster processing with Real-ESRGAN?
Edit `app.py` and add Real-ESRGAN integration (see `INTEGRATION.md`)

---

## Next Steps

See **INTEGRATION.md** for how to integrate Real-ESRGAN and GFPGAN for GPU-accelerated processing.

