import tensorflow as tf

@tf.function(reduce_retracing=True)
def mahalanobis_tf(batch: tf.Tensor, cov_matrices: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Mahalanobis distance for a batch of data using TensorFlow.
    
    Parameters:
    batch (tf.Tensor): A 4D tensor of shape [b, h, w, c] where b is the batch size, 
                       h and w are the height and width of the image, 
                       and c is the number of channels.
    cov_matrices (tf.Tensor): A 3D tensor of shape [b, num_features, num_features]
                              representing the covariance matrices for each item in the batch.
    
    Returns:
    tf.Tensor: A 2D tensor containing Mahalanobis distances for each item in the batch.
    """
    
    # Flatten the height and width dimensions to treat each pixel as a feature
    batch_size = tf.shape(batch)[0]
    channels = tf.shape(batch)[-1]
    batch_flattened = tf.reshape(batch, [batch_size, -1, channels])

    # Cast covariance matrices to tf.float64
    cov_matrices = tf.cast(cov_matrices, tf.float64)

    # Perform Cholesky decomposition
    cholesky_decompositions = tf.linalg.cholesky(cov_matrices)

    # Compute the mean of each batch item
    means = tf.reduce_mean(batch_flattened, axis=1)

    # Calculate Mahalanobis distance using Cholesky decomposition
    mahalanobis_distances = tf.map_fn(
        lambda x: calc_mahalanobis_distance_cholesky(x[0], x[1], x[2]), 
        (batch_flattened, means, cholesky_decompositions), dtype=tf.float64
    )

    return mahalanobis_distances

@tf.function(reduce_retracing=True)
def calc_mahalanobis_distance_cholesky(x: tf.Tensor, mean: tf.Tensor, cholesky_matrix: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Mahalanobis distance for a tensor using the Cholesky decomposition of the covariance matrix.
    
    Parameters:
    x (tf.Tensor): A 2D tensor of shape [num_samples, num_features].
    mean (tf.Tensor): A 1D tensor representing the mean of the features.
    cholesky_matrix (tf.Tensor): A 2D tensor representing the lower triangular Cholesky decomposition of the covariance matrix.
    
    Returns:
    tf.Tensor: A 1D tensor of Mahalanobis distances for each sample.
    """
    diff = x - mean
    diff = tf.transpose(diff, perm=[1, 0])  # Reshape to match expected shape for triangular solve
    y = tf.linalg.triangular_solve(cholesky_matrix, diff, lower=True)
    mahalanobis_distance = tf.sqrt(tf.reduce_sum(tf.square(y), axis=0))
    return mahalanobis_distance
