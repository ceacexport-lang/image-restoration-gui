# Old Photo Restoration Studio

A desktop GUI for restoring old colored photos with local processing and optional remote AI.

## Features

- Load single image or entire folder
- 2x / 4x upscaling
- Denoise, color boost, face restoration, detail enhancement
- One-click auto-restore preset
- Process folders using selected parameters
- Create video from processed images and play it automatically
- Optional remote API support for heavy processing

## Install

```bash
cd ~/image-restoration-gui
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Workflow

### Single Image
1. Click "Load Image"
2. Adjust parameters or click "Restore Old Photo" for auto-preset
3. Click "Run Enhancement"
4. Click "Save Output" to save the result

### Folder with Video Output
1. Click "Load Folder"
2. Adjust parameters
3. Click "Run Enhancement" (processes all images in the folder)
4. After completion, click "Create & Play Video"
5. An MP4 video is created from the processed images and plays automatically

## Parameters

- **Upscale:** 2x or 4x resolution increase
- **Workflow:** Choose restoration style (Faded Color, Portrait, Modern Look, etc.)
- **Denoise:** Remove grain and noise
- **Color:** Boost faded colors (0.5x to 2.5x)
- **Face:** Light or strong face restoration
- **Detail:** Enhance sharpness and details (0.5x to 2.5x)

## License

MIT
