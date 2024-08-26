import unittest
import tensorflow as tf
import numpy as np
from util import fold, unfold
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestImageReconstruction(unittest.TestCase):

    def setUp(self):
        # Load a sample image
        (image, label), _ = tf.keras.datasets.cifar100.load_data()
        self.image = image[:1]  # Use the first image from CIFAR-100 for demonstration, batch size 1
        self.image = tf.image.convert_image_dtype(self.image, dtype=tf.float32)

        # Define patch size
        self.patch_size = 4

    def test_image_reconstruction(self):
        print()
        for padding in ['VALID', 'SAME']:
            for overlap in [True, False]:
                strides = 2 if overlap else self.patch_size

                logging.info(f"Testing padding={padding}, overlap={'Yes' if overlap else 'No'}")

                # Unfold the image into patches
                patches = unfold(self.image, self.patch_size, strides, padding=padding)

                # Fold the patches back into the original image
                reconstructed_image = fold(patches, tf.keras.backend.int_shape(self.image)[1:3], self.patch_size, strides, padding=padding)

                # Convert images back to numpy arrays for comparison
                original_image = tf.squeeze(self.image).numpy()
                reconstructed_image = tf.squeeze(reconstructed_image).numpy()

                # Assert that the original and reconstructed images are equal within epsilon tolerance
                np.testing.assert_allclose(original_image, reconstructed_image, rtol=0, atol=1e-16, err_msg=f"The original and reconstructed images are not equal within the tolerance for padding={padding} and overlap={'Yes' if overlap else 'No'}.")

    def test_image_display(self):
        for padding in ['VALID', 'SAME']:
            for overlap in [True, False]:
                strides = 2 if overlap else self.patch_size

                logging.info(f"Displaying images for padding={padding}, overlap={'Yes' if overlap else 'No'}")

                # Unfold the image into patches
                patches = unfold(self.image, self.patch_size, strides, padding=padding)

                # Fold the patches back into the original image
                reconstructed_image = fold(patches, tf.keras.backend.int_shape(self.image)[1:3], self.patch_size, strides, padding=padding)

                # Convert back to the original scale (0-255) for visualization
                original_image = tf.squeeze(self.image).numpy()
                original_image = (original_image * 255).astype(np.uint8)

                reconstructed_image = tf.squeeze(reconstructed_image)
                reconstructed_image = (reconstructed_image * 255).numpy().astype(np.uint8)

                # Plot the original and reconstructed images
                plt.figure(figsize=(8, 4))

                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(original_image)
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Reconstructed Image")
                plt.imshow(reconstructed_image)
                plt.axis('off')

                plt.show()

if __name__ == '__main__':
    unittest.main()
