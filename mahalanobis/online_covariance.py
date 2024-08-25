import abc
import tensorflow as tf

class OnlineCovarianceAbstract(abc.ABC):
    """
    An abstract base class for online covariance matrix computation.

    This class defines the common interface for all online covariance classes,
    ensuring that each subclass implements the necessary methods.

    Attributes:
    -----------
    num_features : tf.Tensor
        The number of features (flattened dimensions) in the input data.
    """

    def __init__(self, num_features: int):
        """
        Initializes the OnlineCovarianceAbstract with the number of features.

        Parameters:
        -----------
        num_features : int
            The number of features (flattened dimensions) in the input data.
        """
        self.num_features: tf.Tensor = tf.cast(num_features, tf.float64)

    @abc.abstractmethod
    def update(self, batch_data: tf.Tensor) -> None:
        """
        Updates the running statistics with a new batch of data.

        This method must be implemented by any subclass.

        Parameters:
        -----------
        batch_data : tf.Tensor
            A batch of data points to update the covariance statistics.
        
        Returns:
        --------
        None
        """
        pass

    @abc.abstractmethod
    def get_covariance_matrix(self) -> tf.Tensor:
        """
        Returns the current estimate of the covariance matrix.

        This method must be implemented by any subclass.

        Returns:
        --------
        tf.Tensor
            The covariance matrix of the data processed so far.
        """
        pass

class OnlineCovarianceM1(OnlineCovarianceAbstract):
    """
    A class to compute the online covariance matrix using Welford's algorithm
    in a batched manner, optimized for memory usage.

    Attributes:
    -----------
    num_features : int
        The number of features (flattened dimensions) in the input data.
    mean : tf.Variable
        A TensorFlow variable to store the running mean of the features.
    cov_matrix : tf.Variable
        A TensorFlow variable to store the running covariance matrix.
    count : tf.Variable
        A TensorFlow variable to count the number of batches processed.
    """

    def __init__(self, num_features: int):
        """
        Initializes the OnlineCovarianceWelford with the number of features.

        Parameters:
        -----------
        num_features : int
            The number of features (flattened dimensions) in the input data.
        """
        super(OnlineCovarianceM1, self).__init__(num_features=num_features)
        self.mean: tf.Variable = tf.Variable(tf.zeros([num_features], dtype=tf.float64))
        self.cov_matrix: tf.Variable = tf.Variable(tf.zeros([num_features, num_features], dtype=tf.float64))
        self.count: tf.Variable = tf.Variable(0, dtype=tf.int64)

    @tf.function(reduce_retracing=True)
    def update(self, batch_data: tf.Tensor) -> None:
        """
        Updates the running mean and covariance matrix with a new batch of data.

        Parameters:
        -----------
        batch_data : tf.Tensor
            A batch of data with shape [batch_size, height, width, channels].
            The data should be of dtype `float32` and will be cast to `float64`.
        
        Returns:
        --------
        None
        """
        batch_data: tf.Tensor = tf.cast(batch_data, tf.float64)

        # Reshape and flatten the batch data to [batch_size, num_features]
        batch_size: tf.Tensor = tf.cast(tf.shape(batch_data)[0], tf.int64)
        flat_data: tf.Tensor = tf.reshape(batch_data, [batch_size, self.num_features])

        # Update the count with the number of samples in the batch
        new_count: tf.Tensor = self.count + tf.cast(x=batch_size, dtype=tf.int64)

        # Compute the difference between current batch data and the running mean
        delta: tf.Tensor = flat_data - self.mean

        # Update the overall mean with the new batch data
        mean_update: tf.Tensor = tf.reduce_sum(delta, axis=0) / tf.cast(new_count, tf.float64)
        updated_mean: tf.Tensor = self.mean + mean_update

        # Compute the updated delta with the new mean
        delta2: tf.Tensor = flat_data - updated_mean

        # Update the covariance matrix incrementally using the new mean
        cov_update: tf.Tensor = tf.matmul(delta, delta2, transpose_a=True)

        # Apply updates in-place to avoid additional memory allocation
        self.cov_matrix.assign_add(cov_update)
        self.mean.assign(updated_mean)
        self.count.assign(new_count)

    @tf.function(reduce_retracing=True)
    def get_covariance_matrix(self) -> tf.Tensor:
        """
        Returns the current estimate of the covariance matrix.

        Returns:
        --------
        tf.Tensor
            The covariance matrix of the data processed so far. If fewer than
            two batches have been processed, returns a zero matrix.
        """
        if self.count > 1:
            return self.cov_matrix / tf.cast(self.count - 1, tf.float64)
        else:
            return tf.zeros([self.num_features, self.num_features], dtype=tf.float64)

class OnlineCovarianceM2(OnlineCovarianceAbstract):
    """
    This class calculates the covariance matrix of a dataset in an online manner.
    It is useful when the entire dataset cannot fit into memory at once, allowing
    the covariance matrix to be updated incrementally as new data batches are processed.

    Attributes:
    -----------
    num_features : tf.Tensor
        The dimensionality of the data (number of features). This is the number of
        features (or variables) that each data point in the dataset contains.
    n : tf.Variable
        A TensorFlow variable that keeps track of the number of data points processed so far.
        This variable is updated every time a new batch of data is added.
    sum_x : tf.Variable
        A TensorFlow variable that stores the running sum of all the data points
        processed so far. It is used to calculate the mean of the dataset on-the-fly.
    sum_x2 : tf.Variable
        A TensorFlow variable that stores the running sum of the outer products of the
        data points processed so far. It is used to calculate the covariance matrix
        on-the-fly.
    """

    def __init__(self, num_features: int):
        """
        Initialize the OnlineCovarianceM2 object with the number of features.

        Parameters:
        -----------
        num_features : int
            The number of features (or variables) that each data point contains. This value
            is used to initialize the size of the tensors used to store the running statistics.
        """
        super(OnlineCovarianceM2, self).__init__(num_features=num_features)
        
        # Initialize the count of data points processed so far (n)
        self.n: tf.Variable = tf.Variable(0.0, dtype=tf.float64)
        
        # Initialize the running sum of data points (sum_x)
        # This tensor will accumulate the sum of all data points across all batches
        self.sum_x: tf.Variable = tf.Variable(tf.zeros([num_features], dtype=tf.float64), dtype=tf.float64)
        
        # Initialize the running sum of the outer products of data points (sum_x2)
        # This tensor will accumulate the sum of the outer products (x * x^T) of all data points
        self.sum_x2: tf.Variable = tf.Variable(tf.zeros([num_features, num_features], dtype=tf.float64), dtype=tf.float64)

    def update(self, batch_data: tf.Tensor) -> None:
        """
        Update the running statistics (mean and sum of outer products) with a new batch of data.

        This method updates the running statistics used to compute the covariance matrix 
        as new data becomes available. The statistics are updated in-place, so the memory 
        usage is optimized even for large datasets.

        Parameters:
        -----------
        batch_data : tf.Tensor
            A batch of data points with shape [batch_size, num_features]. The data points in 
            the batch should be of dtype `float32` or `float64`. If the data points are not of 
            dtype `float64`, they will be cast to `float64` to ensure numerical precision.
        
        Returns:
        --------
        None
        """
        # Ensure the batch data is of type float64 for consistent numerical precision
        batch_data: tf.Tensor = tf.cast(batch_data, tf.float64)

        # Determine the size of the current batch
        batch_size: tf.Tensor = tf.cast(tf.shape(batch_data)[0], tf.float64)
        
        # Update the count of data points processed so far
        # This adds the number of data points in the current batch to the running total
        self.n.assign_add(batch_size)

        # Update the running sum of the data points
        # This adds the sum of the current batch's data points to the running total
        self.sum_x.assign_add(tf.reduce_sum(batch_data, axis=0))

        # Update the running sum of the outer products of the data points
        # This adds the sum of the outer products (x * x^T) of the current batch's data points to the running total
        self.sum_x2.assign_add(tf.linalg.matmul(tf.transpose(batch_data), batch_data))

    def get_covariance_matrix(self) -> tf.Tensor:
        """
        Calculate and return the covariance matrix of the data processed so far.

        The covariance matrix is computed based on the running statistics that have been 
        accumulated from all the data points processed so far. This method can be called 
        at any time to get the current estimate of the covariance matrix.

        The covariance matrix is calculated using the formula:
        cov(X) = (1/N) * sum_x2 - (mean * mean^T)
        where:
        - N is the number of data points processed.
        - sum_x2 is the running sum of outer products of the data points.
        - mean is the average of the data points, computed as sum_x / N.

        Returns:
        --------
        tf.Tensor
            A tensor representing the covariance matrix of the data processed so far. 
            The shape of this tensor is [num_features, num_features]. If no data has been 
            processed yet, this will return a matrix of zeros.
        """
        # Calculate the mean of the data points processed so far
        mean: tf.Tensor = self.sum_x / self.n
        
        # Calculate the covariance matrix using the running statistics
        cov: tf.Tensor = self.sum_x2 / self.n - tf.tensordot(mean, mean, axes=0)
        
        return cov

class OnlineCovarianceM3(OnlineCovarianceAbstract):
    """
    This class calculates the covariance of a dataset in an online manner using
    a method that aims to improve accuracy by incrementally updating the mean 
    and the sum of squares of differences from the mean.

    Attributes:
    -----------
    num_features : int
        The dimensionality of the data (number of features).
    n : tf.Variable
        The number of data points processed so far.
    mean : tf.Variable
        A TensorFlow variable to store the running mean of the features.
    m2 : tf.Variable
        A TensorFlow variable to store the running sum of squares of differences from the mean.
    """

    def __init__(self, num_features: int):
        """
        Initialize the OnlineCovariance object.

        Parameters:
        -----------
        num_features : int
            The number of features (flattened dimensions) in the input data.
        """
        super(OnlineCovarianceM3, self).__init__(num_features=num_features)
        self.n: tf.Variable = tf.Variable(0.0, dtype=tf.float64)  # The number of data points seen so far.
        self.mean: tf.Variable = tf.Variable(tf.zeros([num_features], dtype=tf.float64))  # The running mean of the data points.
        self.m2: tf.Variable = tf.Variable(tf.zeros([num_features, num_features], dtype=tf.float64))  # The running sum of squares of differences from the mean.

    @tf.function(reduce_retracing=True)
    def update(self, batch_data: tf.Tensor) -> None:
        """
        Update the running statistics (mean and sum of squared deviations) with a new batch of data.

        Parameters:
        -----------
        batch_data : tf.Tensor
            A batch of data points. The shape should be [batch_size, num_features].
            The data should be of dtype `float32` and will be cast to `float64`.

        Returns:
        --------
        None
        """
        batch_data: tf.Tensor = tf.cast(batch_data, tf.float64)

        # Batch size
        batch_size: tf.Tensor = tf.cast(tf.shape(batch_data)[0], tf.float64)

        # Update the count of data points processed so far
        new_n: tf.Tensor = self.n + batch_size

        # Calculate the mean of the batch
        batch_mean: tf.Tensor = tf.reduce_mean(batch_data, axis=0)

        # Update the mean
        delta: tf.Tensor = batch_mean - self.mean
        new_mean: tf.Tensor = self.mean + delta * batch_size / new_n

        # Update the sum of squared deviations from the mean (M2)
        centered_data: tf.Tensor = batch_data - new_mean
        m2_update: tf.Tensor = tf.matmul(centered_data, centered_data, transpose_a=True)
        new_m2: tf.Tensor = self.m2 + m2_update

        # Assign updated values back to the object's state
        self.mean.assign(new_mean)
        self.m2.assign(new_m2)
        self.n.assign(new_n)

    @tf.function(reduce_retracing=True)
    def get_covariance_matrix(self) -> tf.Tensor:
        """
        Calculate and return the covariance matrix of the data seen so far.

        Returns:
        --------
        tf.Tensor
            The covariance matrix. The shape is [num_features, num_features].
        """
        if self.n > 1:
            return self.m2 / (self.n - 1)
        else:
            return tf.zeros([self.num_features, self.num_features], dtype=tf.float64)

class OnlineCovarianceM4(OnlineCovarianceAbstract):
    """
    This class calculates the covariance of a dataset in an online manner using
    a method that aims to improve accuracy by incrementally updating the mean 
    and the sum of squares of differences from the mean.

    Attributes:
    -----------
    num_features : int
        The dimensionality of the data (number of features).
    n : tf.Variable
        The number of data points processed so far.
    mean : tf.Variable
        A TensorFlow variable to store the running mean of the features.
    m2 : tf.Variable
        A TensorFlow variable to store the running sum of squares of differences from the mean.
    """

    def __init__(self, num_features: int):
        """
        Initialize the OnlineCovariance object.

        Parameters:
        -----------
        num_features : int
            The number of features (flattened dimensions) in the input data.
        """
        super(OnlineCovarianceM4, self).__init__(num_features=num_features)
        self.n: tf.Variable = tf.Variable(0.0, dtype=tf.float64)  # The number of data points seen so far.
        self.mean: tf.Variable = tf.Variable(tf.zeros([num_features], dtype=tf.float64))  # The running mean of the data points.
        self.m2: tf.Variable = tf.Variable(tf.zeros([num_features, num_features], dtype=tf.float64))  # The running sum of squares of differences from the mean.

    @tf.function(reduce_retracing=True)
    def update(self, batch_data: tf.Tensor) -> None:
        """
        Update the running statistics (mean and sum of squared deviations) with a new batch of data.

        Parameters:
        -----------
        batch_data : tf.Tensor
            A batch of data points. The shape should be [batch_size, num_features].
            The data should be of dtype `float32` and will be cast to `float64`.

        Returns:
        --------
        None
        """
        batch_data: tf.Tensor = tf.cast(batch_data, tf.float64)

        # Batch size
        batch_size: tf.Tensor = tf.cast(tf.shape(batch_data)[0], tf.float64)

        # Update the count of data points processed so far
        new_n: tf.Tensor = self.n + batch_size

        # Calculate the mean of the batch
        batch_mean: tf.Tensor = tf.reduce_mean(batch_data, axis=0)

        # Update the mean using the current batch mean
        delta: tf.Tensor = batch_mean - self.mean
        new_mean: tf.Tensor = self.mean + delta * batch_size / new_n

        # Update the sum of squared deviations from the mean (M2)
        centered_data: tf.Tensor = batch_data - new_mean
        m2_update: tf.Tensor = tf.matmul(centered_data, centered_data, transpose_a=True)
        new_m2: tf.Tensor = self.m2 + m2_update

        # Assign updated values back to the object's state
        self.mean.assign(new_mean)
        self.m2.assign(new_m2)
        self.n.assign(new_n)

    @tf.function(reduce_retracing=True)
    def get_covariance_matrix(self) -> tf.Tensor:
        """
        Calculate and return the covariance matrix of the data seen so far.

        Returns:
        --------
        tf.Tensor
            The covariance matrix. The shape is [num_features, num_features].
        """
        if self.n > 1:
            cov_matrix: tf.Tensor = self.m2 / (self.n - 1)
            if tf.equal(tf.math.mod(self.n, 1000), 0):
                cov_matrix = self.recalculate_covariance()
            return cov_matrix
        else:
            return tf.zeros([self.num_features, self.num_features], dtype=tf.float64)

    @tf.function(reduce_retracing=True)
    def recalculate_covariance(self) -> tf.Tensor:
        """
        Recalculates the covariance matrix using the current data, correcting any drift.

        Returns:
        --------
        tf.Tensor
            The recalculated covariance matrix.
        """
        centered_data: tf.Tensor = self.m2 / self.n
        new_cov: tf.Tensor = tf.matmul(centered_data, centered_data, transpose_a=True) / (self.n - 1)
        return new_cov

class OnlineCovarianceM5(OnlineCovarianceAbstract):
    """
    A class to calculate the covariance of a dataset in an online manner, 
    optimized for accuracy and stability using Welford's method.

    Attributes:
    -----------
    num_features : tf.Tensor
        The dimensionality of the data (number of features).
    mean : tf.Tensor
        The running mean of the data points.
    M2 : tf.Tensor
        The running sum of squares of differences from the mean.
    n : tf.Tensor
        The number of data points processed so far.
    """

    def __init__(self, num_features: int):
        """
        Initializes the OnlineCovarianceM5 with the number of features.

        Parameters:
        -----------
        num_features : int
            The number of features (flattened dimensions) in the input data.
        """
        super(OnlineCovarianceM5, self).__init__(num_features=num_features)
        self.mean: tf.Tensor = tf.Variable(tf.zeros(num_features, dtype=tf.float64), dtype=tf.float64)
        self.M2: tf.Tensor = tf.Variable(tf.zeros((num_features, num_features), dtype=tf.float64), dtype=tf.float64)
        self.n: tf.Variable = tf.Variable(0.0, dtype=tf.float64)

    def update(self, batch_data: tf.Tensor) -> None:
        """
        Updates the running statistics (mean and sum of squared deviations) with a new batch of data.

        Parameters:
        -----------
        batch_data : tf.Tensor
            A batch of data points. The shape should be [batch_size, num_features].
            The data should be of dtype `float32` and will be cast to `float64`.

        Returns:
        --------
        None
        """
        batch_data: tf.Tensor = tf.cast(batch_data, tf.float64)

        batch_data = tf.convert_to_tensor(batch_data)
        assert batch_data.shape[1] == self.num_features
        batch_size: tf.Tensor = tf.shape(batch_data)[0]
        for i in range(batch_size):
            x: tf.Tensor = batch_data[i]
            self.n.assign_add(1)
            delta: tf.Tensor = x - self.mean
            self.mean.assign_add(delta / self.n)
            delta2: tf.Tensor = x - self.mean
            self.M2.assign_add(tf.tensordot(delta, delta2, axes=0))

    def get_covariance_matrix(self) -> tf.Tensor:
        """
        Returns the current estimate of the covariance matrix.

        Returns:
        --------
        tf.Tensor
            The covariance matrix of the data processed so far. 
            If fewer than two batches have been processed, returns a zero matrix.
        """
        return self.M2 / (self.n - 1) if self.n > 1 else self.M2


# Alias for the OnlineCovarianceM1 class
OnlineCovariance = OnlineCovarianceM1
