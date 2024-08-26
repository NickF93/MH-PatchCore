
import tensorflow as tf
import numpy as np
from util import fold, unfold
import matplotlib.pyplot as plt

# Load a sample image
(image, label), _ = tf.keras.datasets.cifar100.load_data()
image = image[:1]  # Use the first image from CIFAR-100 for demonstration, batch size 1
image = tf.image.convert_image_dtype(image, dtype=tf.float32)

# Define patch size and strides
patch_size = 4
strides = 2

# Unfold the image into patches
patches = unfold(image, patch_size, strides, padding='VALID')
print("Patches shape:", patches.shape)

# Fold the patches back into the original image
reconstructed_image = fold(patches, tf.keras.backend.int_shape(image)[1:3], patch_size, strides, padding='VALID')
print("Reconstructed image shape:", reconstructed_image.shape)

# Convert back to the original scale (0-255) for visualization
original_image = tf.squeeze(image).numpy()
original_image = (original_image * 255).astype(np.uint8)

reconstructed_image = tf.squeeze(reconstructed_image)
reconstructed_image = (reconstructed_image * 255).numpy().astype(np.uint8)

# Extract the first 10 patches and reshape them for visualization
first_10_patches = patches[0, 200:210]  # Select the first 10 patches
first_10_patches = tf.reshape(first_10_patches, [-1, patch_size, patch_size, 3])
first_10_patches = (first_10_patches * 255).numpy().astype(np.uint8)

# Plot the original, reconstructed images, and the first 10 patches
plt.figure(figsize=(15, 8))

plt.subplot(2, 6, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(2, 6, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_image)
plt.axis('off')

# Plot the first 10 patches
for i in range(10):
    plt.subplot(2, 6, i + 3)
    plt.title(f"Patch {i + 1}")
    plt.imshow(first_10_patches[i])
    plt.axis('off')

plt.show()


