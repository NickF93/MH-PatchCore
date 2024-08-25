import unittest
import tensorflow as tf
import numpy as np
from mahalanobis import CovCalc

class TestOnlineCovarianceWelford(unittest.TestCase):

    def test_covariance_matrix(self):
        # Set the epsilon for floating point comparison
        epsilon = 1e-6

        shape = [7, 7, 128]

        # Generate a random tensor of shape [16, 7, 7, 1024]
        data = tf.random.normal(shape=[16, *shape], dtype=tf.float32)

        # Reshape and flatten the tensor to [batch_size, num_features]
        flat_data = tf.reshape(data, [16, -1])

        # Calculate the covariance matrix directly using NumPy
        np_mean = np.mean(flat_data.numpy(), axis=0)
        np_cov_matrix = np.cov(flat_data.numpy(), rowvar=False)

        # Initialize the OnlineCovarianceWelford class
        num_features = int(np.asarray(shape).prod())
        cov_estimator = CovCalc(num_features)

        # Split the data into 4 pieces and update the covariance estimator incrementally
        for i in range(4):
            start_idx = i * 4
            end_idx = (i + 1) * 4
            cov_estimator.update(flat_data[start_idx:end_idx])

        # Get the covariance matrix from the OnlineCovarianceWelford class
        tf_cov_matrix = cov_estimator.get_covariance_matrix().numpy()

        # Compare the matrices element-wise
        np.testing.assert_allclose(tf_cov_matrix, np_cov_matrix, rtol=epsilon, atol=epsilon)

if __name__ == '__main__':
    unittest.main()
