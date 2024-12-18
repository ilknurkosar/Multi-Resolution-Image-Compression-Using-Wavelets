import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from skimage import io
import pywt
from skimage.transform import resize

# -----------------------------------------------------------------------------------------------------------

# Optimize parameters
threshold=0.1
quant_step=5
level=3

# Function to apply Multi-level wavelet compression with Quantization
def wavelet_compression(image, wavelet='bior1.3', threshold=0.1, quant_step=5, level=3):
    coeffs_all_channels = []
    for i in range(image.shape[2]):
        coeffs = pywt.wavedec2(image[:, :, i], wavelet, level=level)
        compressed_coeffs = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                # For detail coefficients (LH,HL,HH) quantize each component
                compressed_coeffs.append(tuple(np.round(c / quant_step) for c in coeff))
            else:
                # For approximation coefficients (LL)
                compressed_coeffs.append(np.round(coeff / quant_step))

        coeffs_all_channels.append(compressed_coeffs)
    return coeffs_all_channels

# Multi-level wavelet decompression
def inverse_wavelet_compression(coeffs_all_channels, wavelet='bior1.3', quant_step=5, original_shape=None):
    reconstructed_image = np.zeros(original_shape, dtype=np.float32)
    for i, coeffs in enumerate(coeffs_all_channels):
        dequantized_coeffs = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                dequantized_coeffs.append(tuple(c * quant_step for c in coeff))
            else:
                dequantized_coeffs.append(coeff * quant_step)
        
        reconstructed_channel = pywt.waverec2(dequantized_coeffs, wavelet)
        reconstructed_image[:, :, i] = reconstructed_channel[:original_shape[0], :original_shape[1]]
    
    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# Function to evaluate PSNR
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# ------------------------------------------------------------------------------------------------------------

path = kagglehub.dataset_download("adityachandrasekhar/image-super-resolution")
print("Path to dataset files:", path)

raw_image_dir = os.path.join(path, "dataset", "Raw Data", "high_res")
image_files_of_high_res = [f for f in os.listdir(raw_image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

# --------------------------------------------------------------------------------------------------------------

image_path = os.path.join(raw_image_dir, image_files_of_high_res[0])
image = io.imread(image_path)

# Store the original shape for later use
original_shape = image.shape

# Apply wavelet compression to the image
compressed_coeffs = wavelet_compression(image, wavelet='bior1.3', threshold=threshold, quant_step=quant_step, level=level)

# Reconstruct the image from compressed coefficients
image_reconstructed = inverse_wavelet_compression(compressed_coeffs, wavelet='bior1.3', quant_step=quant_step, original_shape=original_shape)

# Display the original and reconstructed images
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(image_reconstructed)
plt.show()

# Calculate PSNR (Peak Signal-to-Noise Ratio) between original and reconstructed image
psnr_value = psnr(image, image_reconstructed)
print(f"PSNR (dB) between original and reconstructed image: {psnr_value}")

# Function to visualize wavelet decomposition and reconstruction at each level with PSNR
def visualize_levels_with_psnr(image, wavelet='bior1.3', threshold=0.1, quant_step=5, level=3):
    coeffs_all_channels = []
    for i in range(image.shape[2]):
        coeffs = pywt.wavedec2(image[:, :, i], wavelet, level=level)
        coeffs_all_channels.append(coeffs)

    plt.figure(figsize=(15, 5 * level))

    # Reconstruct at each level by zeroing out higher-level coefficients
    for lvl in range(1, level + 1):
        reconstructed_image = np.zeros(image.shape, dtype=np.float32)
        for i in range(image.shape[2]):
            coeffs = coeffs_all_channels[i]
            reduced_coeffs = []
            for j, c in enumerate(coeffs):
                if j < lvl:
                    reduced_coeffs.append(c)
                else:
                    if isinstance(c, tuple):
                        reduced_coeffs.append(tuple(np.zeros_like(band) for band in c))
                    else:
                        reduced_coeffs.append(np.zeros_like(c))
            
            reconstructed_channel = pywt.waverec2(reduced_coeffs, wavelet)
            reconstructed_channel = reconstructed_channel[:image.shape[0], :image.shape[1]]
            reconstructed_image[:, :, i] = reconstructed_channel

        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
        
        psnr_value = psnr(image, reconstructed_image)
        print(f"PSNR at Level {lvl}: {psnr_value:.2f} dB")

        plt.subplot(level, 1, lvl)
        plt.title(f"Reconstructed at Level {lvl} (PSNR: {psnr_value:.2f} dB)")
        plt.imshow(reconstructed_image)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_levels_with_psnr(image, wavelet='bior1.3', threshold=threshold, quant_step=quant_step, level=level)