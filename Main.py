import os
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from skimage import io
import pywt
from skimage.transform import resize

# -----------------------------------------------------------------------------------------------------------

# Function to apply DWT and compress the image
def wavelet_compression(image, wavelet='bior1.3', threshold=0.1):
    coeffs2 = []
    for i in range(image.shape[2]):
        channel = image[:,:,i]
        coeffs2.append(pywt.dwt2(channel, wavelet))  # Apply DWT to each channel

    compressed_coeffs = []
    for i, (LL, (LH, HL, HH)) in enumerate(coeffs2):
        LL = np.where(np.abs(LL) < threshold, 0, LL)
        LH = np.where(np.abs(LH) < threshold, 0, LH)
        HL = np.where(np.abs(HL) < threshold, 0, HL)
        HH = np.where(np.abs(HH) < threshold, 0, HH)
        compressed_coeffs.append((LL, (LH, HL, HH)))

    return compressed_coeffs

# Function to perform inverse DWT and reconstruct the image
def inverse_wavelet_compression(compressed_coeffs, wavelet='bior1.3', original_shape=None):
    reconstructed_image = np.zeros(original_shape, dtype=np.float32)
    for i in range(original_shape[2]):
        LL, (LH, HL, HH) = compressed_coeffs[i]
        reconstructed_channel = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
        reconstructed_image[:,:,i] = reconstructed_channel

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
compressed_coeffs = wavelet_compression(image, wavelet='bior1.3', threshold=0.1)

# Reconstruct the image from compressed coefficients
image_reconstructed = inverse_wavelet_compression(compressed_coeffs, wavelet='bior1.3', original_shape=original_shape)

# Resizing the reconstructed image to half its original size
half_res_image = resize(image_reconstructed, (original_shape[0] // 2, original_shape[1] // 2), anti_aliasing=True)

# Expanding the resized image back to the original size
expanded_image = resize(half_res_image, (original_shape[0], original_shape[1]), anti_aliasing=True)

# Display the original, resized (half size), and expanded images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.title("Resized (Half Size) Image")
plt.imshow(half_res_image)
plt.subplot(1, 3, 3)
plt.title("Expanded Image")
plt.imshow(expanded_image)
plt.show()

# Calculate PSNR (Peak Signal-to-Noise Ratio) between original and expanded image
psnr_value = psnr(image, expanded_image)
print(f"PSNR (dB) between original and expanded image: {psnr_value}")
