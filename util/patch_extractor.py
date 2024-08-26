import tensorflow as tf

def unfold(image, patch_size, strides):
    """
    Extracts patches from an image.

    Args:
    image: Tensor of shape [b, h, w, c].
    patch_size: Integer, the size of each patch.
    strides: Integer, the stride between patches.

    Returns:
    Tensor of shape [b, p, h, w, c], where p is the number of patches.
    """
    # Extract patches
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, strides, strides, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # Reshape patches to [b, p, h, w, c]
    batch_size = tf.shape(patches)[0]
    patches = tf.reshape(patches, [batch_size, -1, patch_size, patch_size, image.shape[-1]])
    
    return patches

def fold(patches, image_shape, patch_size, strides):
    """
    Reconstructs the image from patches.

    Args:
    patches: Tensor of shape [b, p, h, w, c].
    image_shape: TensorShape, the shape of the original image [b, h, w, c].
    patch_size: Integer, the size of each patch.
    strides: Integer, the stride between patches.

    Returns:
    Tensor of shape [b, h, w, c], the reconstructed image.
    """
    batch_size = image_shape[0]
    height = image_shape[1]
    width = image_shape[2]
    channels = image_shape[3]

    # Initialize an empty tensor to hold the reconstructed image
    reconstructed_image = tf.zeros(image_shape, dtype=patches.dtype)
    overlap_count = tf.zeros(image_shape, dtype=patches.dtype)

    # Calculate number of patches along height and width
    num_patches_h = (height - patch_size) // strides + 1
    num_patches_w = (width - patch_size) // strides + 1

    # Iterate over the patches and add them to the reconstructed image
    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = patches[:, patch_idx, :, :, :]
            patch = tf.reshape(patch, [batch_size, patch_size, patch_size, channels])

            # Calculate the top-left corner where the patch should be placed
            h_start = i * strides
            w_start = j * strides

            # Generate the indices for where the patch should be added
            batch_indices = tf.range(batch_size)
            h_indices = tf.range(h_start, h_start + patch_size)
            w_indices = tf.range(w_start, w_start + patch_size)
            c_indices = tf.range(channels)

            # Create a meshgrid for indices
            b, h, w, c = tf.meshgrid(batch_indices, h_indices, w_indices, c_indices, indexing='ij')
            indices = tf.stack([b, h, w, c], axis=-1)

            # Reshape indices and updates to match
            indices = tf.reshape(indices, [-1, 4])
            updates = tf.reshape(patch, [-1])

            # Update the reconstructed image
            reconstructed_image = tf.tensor_scatter_nd_add(reconstructed_image, indices, updates)

            # Update the overlap count for normalization
            overlap_patch = tf.ones_like(patch)
            overlap_updates = tf.reshape(overlap_patch, [-1])
            overlap_count = tf.tensor_scatter_nd_add(overlap_count, indices, overlap_updates)

            patch_idx += 1

    # Normalize the reconstructed image where patches overlap
    reconstructed_image = tf.math.divide_no_nan(reconstructed_image, overlap_count)

    return reconstructed_image

def tf_unfold(input_tensor, kernel_size, stride, padding='VALID'):
    # Assume input_tensor is in NHWC format: [batch_size, height, width, channels]
    batch_size, height, width, in_channels = input_tensor.shape
    kernel_height, kernel_width = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)

    # Extract patches
    patches = tf.image.extract_patches(
        images=input_tensor,
        sizes=[1, kernel_height, kernel_width, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding=padding.upper()  # Use TensorFlow's padding format
    )

    # Reshape patches to match the desired output shape [batch_size, num_patches, patch_height, patch_width, channels]
    patches = tf.reshape(patches, [batch_size, -1, kernel_height, kernel_width, in_channels])

    return patches

def tf_fold(patches, output_size, kernel_size, stride, padding='SAME'):
    batch_size, num_patches, patch_height, patch_width, in_channels = patches.shape
    output_height, output_width = output_size
    kernel_height, kernel_width = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
    
    # Calculate the effective height and width after padding
    if padding == 'SAME':
        pad_total_height = int(max((tf.math.ceil(output_height / stride) - 1) * stride + kernel_height - output_height, 0))
        pad_total_width = int(max((tf.math.ceil(output_width / stride) - 1) * stride + kernel_width - output_width, 0))

        pad_top = pad_total_height // 2
        pad_bottom = pad_total_height - pad_top
        pad_left = pad_total_width // 2
        pad_right = pad_total_width - pad_left
    else:
        pad_top = pad_left = pad_bottom = pad_right = pad_total_height = pad_total_width = 0

    # Initialize the output tensor and a count tensor to keep track of the overlapping regions
    output_tensor = tf.zeros([batch_size, output_height + pad_total_height, output_width + pad_total_width, in_channels], dtype=patches.dtype)
    count_tensor = tf.zeros([batch_size, output_height + pad_total_height, output_width + pad_total_width, in_channels], dtype=patches.dtype)

    idx = 0
    for i in range(0, output_height - kernel_height + 1 + pad_top + pad_bottom, stride):
        for j in range(0, output_width - kernel_width + 1 + pad_left + pad_right, stride):
            patch = patches[:, idx, :, :, :]  # Extract the current patch

            # Calculate indices for the current patch position
            batch_indices = tf.range(batch_size)[:, None, None, None]
            height_indices = tf.range(i, i + patch_height)[None, :, None, None]
            width_indices = tf.range(j, j + patch_width)[None, None, :, None]
            channel_indices = tf.range(in_channels)[None, None, None, :]

            indices = tf.stack(tf.meshgrid(batch_indices, height_indices, width_indices, channel_indices, indexing='ij'), axis=-1)
            indices = tf.reshape(indices, [-1, 4])

            # Flatten the patch to match the flattened indices
            flat_patch = tf.reshape(patch, [-1])

            # Scatter add the patch values into the output tensor
            output_tensor = tf.tensor_scatter_nd_add(output_tensor, indices, flat_patch)
            count_tensor = tf.tensor_scatter_nd_add(count_tensor, indices, tf.ones_like(flat_patch))

            idx += 1

    # Normalize the output tensor by the count tensor to handle overlapping areas
    output_tensor = tf.math.divide_no_nan(output_tensor, count_tensor)
    
    # Remove padding from the output tensor
    output_tensor = output_tensor[:, pad_top:pad_top + output_height, pad_left:pad_left + output_width, :]

    return output_tensor
