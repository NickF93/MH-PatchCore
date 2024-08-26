import tensorflow as tf
import enum
import typing

class RescalingTypeEnum(enum.Enum):
    NONE        = 0
    COVARIANCE  = 1
    CHOLESKY    = 2

def rescale_factor(cov_matrix: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.reduce_max(tf.linalg.eigvals(cov_matrix)), tf.float64)

def inv(matrix: tf.Tensor) -> tf.Tensor:
    return tf.linalg.inv(matrix)

def covariance(cov_matrix: tf.Tensor, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    if rescale == RescalingTypeEnum.NONE:
        resc_fct = tf.cast(int(1), tf.float64)
        return cov_matrix, resc_fct
    elif rescale == RescalingTypeEnum.COVARIANCE:
        resc_fct = rescale_factor(cov_matrix)
        return tf.math.divide_no_nan(cov_matrix, resc_fct), resc_fct
    elif rescale == RescalingTypeEnum.CHOLESKY:
        resc_fct = rescale_factor(cov_matrix)
        return cov_matrix, resc_fct

def cholesky(cov_matrix: tf.Tensor, rescale_factor: typing.Optional[tf.Tensor] = None, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    L = tf.linalg.cholesky(cov_matrix)
    if rescale == RescalingTypeEnum.NONE:
        resc_fct = tf.cast(int(1), tf.float64)
        invL = inv(L)
        return L, resc_fct, invL
    elif rescale == RescalingTypeEnum.COVARIANCE:
        resc_fct = rescale_factor if rescale_factor is not None else rescale_factor(cov_matrix)
        invL = inv(L)
        return L, resc_fct, invL
    elif rescale == RescalingTypeEnum.CHOLESKY:
        resc_fct = rescale_factor if rescale_factor is not None else rescale_factor(cov_matrix)
        L = L / resc_fct
        invL = inv(L)
        return L, resc_fct, invL

def format_s1s2C(s1: tf.Tensor, s2: tf.Tensor, cov_matrix: tf.Tensor, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    C, rescale_factor = covariance(cov_matrix, rescale)
    
    c1 = tf.shape(s1)[-1]
    c2 = tf.shape(s2)[-1]
    assert c1 == c2, 's1 and s2 must have the same channel size'

    c = tf.cast(c1, tf.int64)
    del c1, c2

    s1 = tf.reshape(s1, (-1, c))
    s2 = tf.reshape(s2, (-1, c))

    n, _ = tf.keras.backend.int_shape(s1)
    b, _ = tf.keras.backend.int_shape(s2)

    return C, s1, s2, c, tf.cast(n, tf.int64), tf.cast(b, tf.int64), rescale_factor

def mahalanobis_matrix_1(s1: tf.Tensor, s2: tf.Tensor, cov_matrix: tf.Tensor, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> tf.Tensor:
    C, s1, s2, c, n, b, rescale_factor = format_s1s2C(s1, s2, cov_matrix, rescale)

    # Step 1: Compute the inverse of the covariance matrix C
    C_inv = tf.linalg.inv(C)

    # Step 2: Compute the Mahalanobis distance between each vector in S1 and each vector in S2
    dist_matrix_1 = tf.zeros([n, b], dtype=tf.float64)

    for i in range(n):
        for j in range(b):
            diff = s1[i] - s2[j]
            dist = tf.sqrt(tf.matmul(tf.matmul(diff[tf.newaxis, :], C_inv), diff[:, tf.newaxis]))
            dist_matrix_1 = tf.tensor_scatter_nd_update(dist_matrix_1, [[i, j]], [dist[0][0]])

    return dist_matrix_1

def mahalanobis_matrix_2(s1: tf.Tensor, s2: tf.Tensor, cov_matrix: tf.Tensor, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> tf.Tensor:
    C, s1, s2, c, n, b, rescale_factor = format_s1s2C(s1, s2, cov_matrix, rescale)

    # Step 1: Compute the Cholesky decomposition of C
    L, rescale_factor, invL = cholesky(cov_matrix, rescale_factor, rescale)

    # Step 2: Compute the Mahalanobis distance using triangular solve
    dist_matrix_2 = tf.zeros([n, b], dtype=tf.float64)

    for i in range(n):
        for j in range(b):
            diff = s1[i] - s2[j]
            w = tf.linalg.triangular_solve(L, diff[:, tf.newaxis], lower=True)
            dist_matrix_2 = tf.tensor_scatter_nd_update(dist_matrix_2, [[i, j]], [tf.sqrt(tf.reduce_sum(w ** 2))])

    return dist_matrix_2

def mahalanobis_matrix_3(s1: tf.Tensor, s2: tf.Tensor, cov_matrix: tf.Tensor, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> tf.Tensor:
    C, s1, s2, c, n, b, rescale_factor = format_s1s2C(s1, s2, cov_matrix, rescale)

    # Step 1: Compute the inverse of the Cholesky decomposition matrix L
    L, rescale_factor, invL = cholesky(cov_matrix, rescale_factor, rescale)

    # Step 2: Compute the Mahalanobis distance using L and L_inv
    dist_matrix_3 = tf.zeros([n, b], dtype=tf.float64)

    for i in range(n):
        for j in range(b):
            diff = s1[i] - s2[j]
            u = tf.matmul(invL, diff[:, tf.newaxis])
            dist_matrix_3 = tf.tensor_scatter_nd_update(dist_matrix_3, [[i, j]], [tf.sqrt(tf.reduce_sum(u ** 2))])

    return dist_matrix_3

def mahalanobis_matrix_3_opt(s1: tf.Tensor, s2: tf.Tensor, cov_matrix: tf.Tensor, rescale: RescalingTypeEnum = RescalingTypeEnum.NONE) -> tf.Tensor:
    # Assuming format_s1s2C and cholesky are utility functions that return formatted inputs.
    C, s1, s2, c, n, b, rescale_factor = format_s1s2C(s1, s2, cov_matrix, rescale)

    # Step 1: Compute the Cholesky decomposition of the covariance matrix and its inverse
    L, rescale_factor, invL = cholesky(cov_matrix, rescale_factor, rescale)

    # Step 2: Compute the difference matrix between s1 and s2 vectors
    # Shape of diff_matrix: [n, b, c]
    s1_expanded = tf.expand_dims(s1, axis=1)  # Shape [n, 1, c]
    s2_expanded = tf.expand_dims(s2, axis=0)  # Shape [1, b, c]
    diff_matrix = s1_expanded - s2_expanded   # Broadcasting to shape [n, b, c]

    # Step 3: Compute the transformed differences u = invL * diff_matrix
    # Since invL is [c, c] and diff_matrix is [n, b, c], we need to perform a batch matrix multiplication
    # Transpose invL for correct broadcasting in matmul
    invL_T = tf.transpose(invL)
    u_matrix = tf.einsum('ij,nkj->nki', invL_T, diff_matrix)  # Shape [n, b, c]

    # Step 4: Compute the Mahalanobis distance using the norm of the u_matrix
    dist_matrix_3 = tf.sqrt(tf.reduce_sum(tf.square(u_matrix), axis=-1))  # Shape [n, b]

    return dist_matrix_3

#@tf.function(reduce_retracing=True)
def mahalanobis_tf(batch: tf.Tensor, cov_matrix: tf.Tensor, rescaling_type: RescalingTypeEnum = RescalingTypeEnum.NONE, rescale_back: bool = True) -> tf.Tensor:
    """
    Calculates the Mahalanobis distance for a batch of data using TensorFlow.
    
    Parameters:
    batch (tf.Tensor): A 4D tensor of shape [b, h, w, c] where b is the batch size, 
                       h and w are the height and width of the image, 
                       and c is the number of channels.
    cov_matrix (tf.Tensor): A 3D tensor of shape [b, num_features, num_features]
                              representing the covariance matrices for each item in the batch.
    
    Returns:
    tf.Tensor: A 2D tensor containing Mahalanobis distances for each item in the batch.
    """
    
    rescale_factor = tf.cast(int(1), tf.float64)
    if rescaling_type == RescalingTypeEnum.NONE:
        rescale_factor = None

    # Flatten the height and width dimensions to treat each pixel as a feature
    batch_size = tf.shape(batch)[0]
    channels = tf.shape(batch)[-1]
    batch_flattened = tf.reshape(batch, [-1, channels])

    # Cast covariance matrices to tf.float64
    cov_matrix = tf.cast(cov_matrix, tf.float64)

    if rescaling_type == RescalingTypeEnum.COVARIANCE:
        # Use max eigenvalue as rescale factor
        rescale_factor = tf.reduce_max(tf.linalg.eigvalsh(cov_matrix))
        cov_matrix = tf.math.divide_no_nan(cov_matrix, rescale_factor)
        rescale_factor = tf.sqrt(rescale_factor)

    # Perform Cholesky decomposition
    cholesky_decompositions = tf.linalg.cholesky(cov_matrix)

    if rescaling_type == RescalingTypeEnum.CHOLESKY:
        rescale_factor = tf.reduce_max(tf.map_fn(lambda L: tf.reduce_max(tf.abs(L)), cholesky_decompositions))
        cholesky_decompositions = tf.math.divide_no_nan(cholesky_decompositions, rescale_factor)

    # Compute the mean of each batch item
    means = tf.reduce_mean(batch_flattened, axis=0)

    batch_flattened_centered = batch_flattened - means

    if not rescale_back:
        rescale_factor = None

    # Calculate Mahalanobis distance using Cholesky decomposition
    batch_flattened_transformed = tf.linalg.triangular_solve(cholesky_decompositions, tf.transpose(batch_flattened_centered), lower=True)
    mahalanobis_distances = tf.norm(batch_flattened_transformed[:, None, :] - batch_flattened_transformed[None, :, :], axis=-1)

    if rescale_factor is not None:
        mahalanobis_distances = tf.math.divide_no_nan(mahalanobis_distances, rescale_factor)

    return mahalanobis_distances
