import tensorflow as tf
import numpy as np
import unittest
from mahalanobis import OnlineMeanStd

class TestOnlineMeanStd(unittest.TestCase):
    def test_online_vs_standard(self):
        num_features = 10

        # Initialize the online mean and std calculator
        online_calculator = OnlineMeanStd(num_features)

        # Generate some test data (3 batches of random data)
        batch1 = np.random.normal(loc=0.0, scale=1.0, size=(100, num_features))
        batch2 = np.random.normal(loc=0.5, scale=1.5, size=(200, num_features))
        batch3 = np.random.normal(loc=-0.5, scale=2.0, size=(150, num_features))

        # Convert batches to TensorFlow tensors
        batch1_tf = tf.convert_to_tensor(batch1, dtype=tf.float64)
        batch2_tf = tf.convert_to_tensor(batch2, dtype=tf.float64)
        batch3_tf = tf.convert_to_tensor(batch3, dtype=tf.float64)

        # Update the online calculator with each batch
        online_calculator.update(batch1_tf)
        online_calculator.update(batch2_tf)
        online_calculator.update(batch3_tf)

        # Finalize to get the online mean and standard deviation
        online_mean, online_stddev = online_calculator.finalize()

        # Standard (batch) calculation of mean and standard deviation
        all_data = np.concatenate([batch1, batch2, batch3], axis=0)
        standard_mean = np.mean(all_data, axis=0)
        standard_stddev = np.std(all_data, axis=0, ddof=0)

        # Convert standard results to TensorFlow tensors for comparison
        standard_mean_tf = tf.convert_to_tensor(standard_mean, dtype=tf.float64)
        standard_stddev_tf = tf.convert_to_tensor(standard_stddev, dtype=tf.float64)

        # Compare the results
        tf.debugging.assert_near(online_mean, standard_mean_tf, atol=1e-7, message="Mean mismatch")
        tf.debugging.assert_near(online_stddev, standard_stddev_tf, atol=1e-7, message="Standard deviation mismatch")

        print("Test passed: Online calculations match standard calculations")

# Run the test
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)