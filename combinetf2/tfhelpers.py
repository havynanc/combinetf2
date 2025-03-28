import tensorflow as tf


def simple_sparse_slice0end(in_sparse, end):
    """
    Slice a tf.sparse.SparseTensor along axis 0 starting 0 to the 'end'.
    """

    # Convert dense_shape, indices, and values to tensors if they aren't already
    dense_shape = in_sparse.dense_shape
    indices = in_sparse.indices
    values = in_sparse.values

    # Compute output dense shape after slicing
    out_shape = tf.concat([[end], dense_shape[1:]], axis=0)

    # Filter rows: select entries where indices[:, 0] < end
    mask = indices[:, 0] < end
    selected_indices = tf.boolean_mask(indices, mask)
    selected_values = tf.boolean_mask(values, mask)

    # Return the sliced sparse tensor
    return tf.sparse.SparseTensor(
        indices=selected_indices, values=selected_values, dense_shape=out_shape
    )


def is_diag(x):
    return tf.math.equal(
        tf.math.count_nonzero(x), tf.math.count_nonzero(tf.linalg.diag_part(x))
    )
