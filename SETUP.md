# Setup and Run Guide

## Install Python

Use Python 3.10+.

## Create virtual environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## Install project dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

## Optional: install Real-ESRGAN and GFPGAN

For stronger AI-based output, especially on a system with 16GB VRAM:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install realesrgan gfpgan facexlib
```

If you do not install these extras, the app will still run and use its built-in enhancement pipeline.

## Recommended presets

### Faded color photo
- Scale: 2x or 4x
- Denoise: enabled
- Color boost: 1.20
- Contrast: 1.10
- Sharpness: 1.20
- Workflow: Faded Color Restore

### Portrait
- Scale: 2x or 4x
- Denoise: enabled
- Face restoration: enabled
- Color boost: 1.15
- Sharpness: 1.25
- Workflow: Portrait Refresh

### Damaged scan
- Scale: 2x
- Denoise: enabled
- Contrast: 1.20
- Sharpness: 1.35
- Workflow: High Detail Restore

## Troubleshooting

### Package install fails

Try:
```bash
pip install --upgrade pip setuptools wheel
```

### GUI does not open

Ensure PyQt5 is installed:
```bash
pip install PyQt5
```

### Real-ESRGAN not available

This is expected if the optional AI packages were not installed. The app will automatically fall back to the built-in enhancement mode.
