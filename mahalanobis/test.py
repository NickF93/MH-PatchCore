import tensorflow as tf

# Assume X is your input tensor of shape [100, 10]
X = tf.random.normal([100, 10])

# Step 1: Compute the covariance matrix
mean_X = tf.reduce_mean(X, axis=0)
X_centered = X - mean_X
cov_matrix = tf.matmul(X_centered, X_centered, transpose_a=True) / (X.shape[0] - 1)

# Step 2: Cholesky decomposition
L = tf.linalg.cholesky(cov_matrix)  # L is [10, 10]

# Step 3: Use triangular solve to compute L^{-1} * (X_centered)
# Since L is [10, 10] and X_centered is [100, 10], we need to transpose X_centered for correct dimensions
X_transformed = tf.linalg.triangular_solve(L, tf.transpose(X_centered), lower=True)

# Step 4: Calculate Mahalanobis distances
# The pairwise Mahalanobis distances are the Euclidean distances in the transformed space
dists = tf.norm(X_transformed[:, None, :] - X_transformed[None, :, :], axis=-1)

# dists is a [100, 100] matrix of Mahalanobis distances between each pair of samples
