import tensorflow as tf
import numpy as np
import unittest
from mahalanobis import CovCalc, mahalanobis

class TestMahalanobisDistance(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(0)
        self.num_samples = 100
        self.num_features = 10
        self.epsilon = 1e-6

        # Generate random data
        self.data = tf.convert_to_tensor(np.random.rand(self.num_samples, self.num_features), dtype=tf.float64)
        
        # Calculate the covariance matrix
        self.cov_matrix = CovCalc(self.num_features)
        self.cov_matrix.update(self.data)
        self.cov_matrix = self.cov_matrix.get_covariance_matrix()

        # Compute mean
        self.mean = tf.reduce_mean(self.data, axis=0)

    def test_mahalanobis_distance(self):
        # Calculate Mahalanobis distance using the provided function
        batch_data = tf.expand_dims(self.data, axis=0)  # Adding batch dimension
        cov_matrices = tf.expand_dims(self.cov_matrix, axis=0)  # Adding batch dimension
        calculated_distances = mahalanobis(batch_data, cov_matrices)
        
        # Calculate Mahalanobis distance using the classic method (inverse covariance matrix)
        inv_cov_matrix = tf.linalg.inv(self.cov_matrix)
        diff = self.data - self.mean
        left = tf.matmul(diff, inv_cov_matrix)
        classic_distances = tf.sqrt(tf.reduce_sum(left * diff, axis=1))
        
        # Compare the distances within an epsilon range
        self.assertTrue(tf.reduce_all(tf.abs(calculated_distances - classic_distances) < self.epsilon),
                        msg=f"Max difference: {tf.reduce_max(tf.abs(calculated_distances - classic_distances))}")

if __name__ == '__main__':
    unittest.main()
