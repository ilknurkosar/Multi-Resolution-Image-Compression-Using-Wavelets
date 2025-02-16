# **Multi-Resolution-Image-Compression-Using-Wavelets**
## 📌 **Project Overview**
This project implements **multi-level discrete wavelet transform (DWT) based image compression** using **Python and PyWavelets**. The goal is to **reduce image size** while preserving key visual details using **quantization and thresholding techniques**. 

The process involves:
- **Wavelet decomposition**: Breaking down an image into different frequency components using `pywt.wavedec2`
- **Quantization**: Reducing precision of wavelet coefficients to improve compression
- **Thresholding**: Eliminating small, insignificant coefficients to reduce storage needs
- **Wavelet reconstruction**: Rebuilding the image using `pywt.waverec2` after compression
- **Performance evaluation**: Measuring the quality of reconstructed images using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**.

---

## 🚀 **Features**
✅ **Multi-Level Wavelet Compression** using `pywt.wavedec2`  
✅ **Quantization-Based Compression** with adjustable `quant_step`  
✅ **Thresholding for Higher Compression**  
✅ **Reconstruction using `pywt.waverec2`**  
✅ **Performance Metrics (PSNR & SSIM)**  
✅ **Data visualization for compressed & reconstructed images**

---

## 🔧 **Technologies Used**
- **Python** (Main programming language)
- **PyWavelets (`pywt`)** for Discrete Wavelet Transform (DWT & IDWT)
- **NumPy (`numpy`)** for array manipulations
- **Matplotlib (`matplotlib`)** for image visualization
- **scikit-image (`skimage.metrics`)** for SSIM calculations

---
