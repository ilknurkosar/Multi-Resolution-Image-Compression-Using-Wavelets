import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from skimage import io
import pywt
from skimage.metrics import structural_similarity as ssim
import time
import tracemalloc

# Optimize Parameters
threshold = 0.1
quant_step = 5
level = 3
# -----------------------------------------------------------------------------------------------------------

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

# Function to evaluate SSIM
def calculate_ssim(original, compressed):
    # Convert images to grayscale if they are RGB
    if original.ndim == 3:
        original_gray = np.mean(original, axis=2)
        compressed_gray = np.mean(compressed, axis=2)
    else:
        original_gray = original
        compressed_gray = compressed

    ssim_value = ssim(original_gray, compressed_gray, data_range=compressed_gray.max() - compressed_gray.min())
    return ssim_value

# Function to visualize wavelet decomposition and reconstruction at each level with PSNR and SSIM
def visualize_levels_with_psnr(image, wavelet_types, threshold=0.1, quant_step=5, level=3):
    psnr_values_all = {wavelet: [] for wavelet in wavelet_types}
    ssim_values_all = {wavelet: [] for wavelet in wavelet_types}

    for wavelet in wavelet_types:
        coeffs_all_channels = []
        for i in range(image.shape[2]):
            coeffs = pywt.wavedec2(image[:, :, i], wavelet, level=level)
            coeffs_all_channels.append(coeffs)

        plt.figure(figsize=(12, 6))

        # Reconstruct at each level by zeroing out higher-level coefficients
        plt.subplot(1, level + 1, 1)
        plt.title(f"Original Image")
        plt.imshow(image)
        plt.axis('off')

        for lvl in range(level):
            reconstructed_image = np.zeros(image.shape, dtype=np.float32)
            for i in range(image.shape[2]):
                coeffs = coeffs_all_channels[i]
                reduced_coeffs = []
                for j, c in enumerate(coeffs):
                    if j <= lvl:
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
            ssim_value = calculate_ssim(image, reconstructed_image)
            psnr_values_all[wavelet].append(psnr_value)
            ssim_values_all[wavelet].append(ssim_value)
            print(f"Wavelet: {wavelet}, Level {lvl + 1}: PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")

           # Add reconstructed image in subsequent plots
            plt.subplot(1, level + 1, lvl + 2)
            plt.title(f"Reconstructed (Level {lvl + 1} {wavelet})\nPSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
            plt.imshow(reconstructed_image)
            plt.axis('off')

        plt.tight_layout()
        
    
        plt.show()


    # Plot PSNR values for each wavelet type
    plt.figure(figsize=(10, 6))
    for wavelet, psnr_values in psnr_values_all.items():
        plt.plot(range(1, level + 1), psnr_values, marker='o', label=f"{wavelet} (PSNR)")

    plt.title("PSNR across Levels for All Wavelet Types")
    plt.xlabel("Level")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

    # Plot SSIM values for each wavelet type
    plt.figure(figsize=(10, 6))
    for wavelet, ssim_values in ssim_values_all.items():
        plt.plot(range(1, level + 1), ssim_values, marker='o', label=f"{wavelet} (SSIM)")

    plt.title("SSIM across Levels for All Wavelet Types")
    plt.xlabel("Level")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

    compressed_coeffs = wavelet_compression(image, wavelet='bior1.3', quant_step=quant_step, level=level)
    calculate_compression_ratio(image, compressed_coeffs, quant_step)
    evaluate_performance(image, wavelet='bior1.3', quant_step=quant_step, level=level)

def calculate_compression_ratio(original_image, compressed_coeffs, quant_step):
    original_size = original_image.nbytes
    
    compressed_size = 0
    for channel in compressed_coeffs:
        for coeff in channel:
            if isinstance(coeff, tuple):
                for sub_coeff in coeff:
                    compressed_size += np.count_nonzero(np.abs(sub_coeff) > quant_step) * sub_coeff.itemsize
            else:
                compressed_size += np.count_nonzero(np.abs(coeff) > quant_step) * coeff.itemsize

    compression_ratio = original_size / compressed_size

    print(f"Original Size: {original_size} bytes")
    print(f"Compressed Size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    
    return compression_ratio


def evaluate_performance(image, wavelet='bior1.3', quant_step=5, level=3):
    tracemalloc.start()
    start_time = time.time()

    compressed_coeffs = wavelet_compression(image, wavelet=wavelet, quant_step=quant_step, level=level)

    end_time = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    compression_time = end_time - start_time
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory / 1024**2:.2f} MB")

    # Measure reconstruction time
    start_time = time.time()

    reconstructed_image = inverse_wavelet_compression(
        compressed_coeffs, wavelet=wavelet, quant_step=quant_step, original_shape=image.shape
    )

    end_time = time.time()
    reconstruction_time = end_time - start_time
    print(f"Reconstruction Time: {reconstruction_time:.2f} seconds")
    return compressed_coeffs, reconstructed_image
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
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(image_reconstructed)
plt.axis('off')


plt.show()
# Calculate PSNR (Peak Signal-to-Noise Ratio) between original and reconstructed image
psnr_value = psnr(image, image_reconstructed)
print(f"PSNR (dB): {psnr_value}")


# List of wavelet types to test
wavelet_types = ['bior1.3', 'haar', 'db2', 'coif1']

# Visualize PSNR values across levels for all wavelet types
visualize_levels_with_psnr(image, wavelet_types, threshold=0.1, quant_step=5, level=3)
