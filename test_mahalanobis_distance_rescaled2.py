import gc
import tensorflow as tf
import numpy as np
import unittest
import time  # Import the time module for measuring execution time
from mahalanobis import CovCalc, mahalanobis_matrix, mahalanobis_matrix_1, mahalanobis_matrix_2, mahalanobis_matrix_3, RescalingTypeEnum

print ("TensorFlow version:", tf.__version__)

gc.collect()
tf.keras.backend.clear_session()
gc.collect()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class TestMahalanobisDistance(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        np.random.seed(0)
        self.n = 100
        self.b = 10
        self.c = 10
        self.epsilon = 1e-6

        # Generate random data
        self.s1 = tf.convert_to_tensor(np.random.rand(self.n, self.c), dtype=tf.float64)
        self.s2 = tf.convert_to_tensor(np.random.rand(self.b, self.c), dtype=tf.float64)

    def test_mahalanobis_distance(self):
        cov_estimator = CovCalc(self.c)
        cov_estimator.update(self.s1)
        cov_matrix = cov_estimator.get_covariance_matrix()
        
        # Timing mahalanobis_matrix_2
        start_time = time.time()
        m2 = mahalanobis_matrix_2(self.s1, self.s2, cov_matrix, rescale=RescalingTypeEnum.CHOLESKY)
        duration_2 = time.time() - start_time
        print(f"mahalanobis_matrix_2 execution time: {duration_2:.6f} seconds")
        
        # Timing mahalanobis_matrix_3
        start_time = time.time()
        m3 = mahalanobis_matrix_3(self.s1, self.s2, cov_matrix, rescale=RescalingTypeEnum.CHOLESKY)
        duration_3 = time.time() - start_time
        print(f"mahalanobis_matrix_3 execution time: {duration_3:.6f} seconds")
        
        # Timing mahalanobis_matrix
        start_time = time.time()
        m4 = mahalanobis_matrix(self.s1, self.s2, cov_matrix, rescale=RescalingTypeEnum.CHOLESKY)
        duration_4 = time.time() - start_time
        print(f"mahalanobis_matrix execution time: {duration_4:.6f} seconds")
        
        # Compare the distances within an epsilon range
        self.assertTrue(tf.reduce_all(tf.abs(m2 - m3) < self.epsilon),
                        msg=f"Max difference: {tf.reduce_max(tf.abs(m2 - m3))}")
        
        # Compare the distances within an epsilon range
        self.assertTrue(tf.reduce_all(tf.abs(m2 - m3) < self.epsilon),
                        msg=f"Max difference: {tf.reduce_max(tf.abs(m2 - m4))}")

if __name__ == '__main__':
    unittest.main()
