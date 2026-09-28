#!/bin/bash
# Complete fresh reinstall script for Old Photo Restoration Studio

set -e  # Exit on any error

echo "========================================"
echo "Old Photo Restoration Studio - Fresh Install"
echo "========================================"
echo ""

cd ~/image-restoration-gui || { echo "Error: ~/image-restoration-gui directory not found"; exit 1; }

echo "[1/8] Removing old virtual environment..."
rm -rf venv
echo "✓ venv removed"
echo ""

echo "[2/8] Creating fresh virtual environment..."
python3 -m venv venv
echo "✓ venv created"
echo ""

echo "[3/8] Activating virtual environment..."
source venv/bin/activate
echo "✓ venv activated"
echo ""

echo "[4/8] Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel
echo "✓ pip tools upgraded"
echo ""

echo "[5/8] Installing base requirements..."
pip install -r requirements.txt
echo "✓ base requirements installed"
echo ""

echo "[6/8] Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch installed"
echo ""

echo "[7/8] Installing AI restoration models (Real-ESRGAN, GFPGAN, FaceXLib)..."
pip install realesrgan gfpgan facexlib
echo "✓ AI models installed"
echo ""

echo "[8/8] Launching application..."
echo "✓ Starting Old Photo Restoration Studio"
echo ""
python app.py
