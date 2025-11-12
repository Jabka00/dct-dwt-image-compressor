# Image Compressor DCT/DWT

Image compression tool using DCT (Discrete Cosine Transform) and DWT (Discrete Wavelet Transform) with PSNR quality assessment.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the program:
```bash
python image_compressor.py
```

1. Click "Завантажити зображення" (Load Image)
2. Adjust parameters:
   - **DCT Quality**: 0-100 (higher = better quality)
   - **DWT Coefficients**: 0-100 (percentage of preserved coefficients)
3. Click "Стиснути" (Compress)
4. Compare results and PSNR values

## Methods

- **DCT**: 8×8 blocks, JPEG quantization matrix
- **DWT**: Haar wavelet, 3-level decomposition
- **PSNR**: Automatic quality calculation in dB

## Requirements

- Python 3.7+
- Grayscale images (recommended 512×512)
