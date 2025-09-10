import tensorflow as tf

@tf.function()
def tx_distance(grid, angles):
    x = tf.expand_dims(grid[..., 0], axis=0)
    z = tf.expand_dims(grid[..., 2], axis=0)

    # Distance along steering direction:  x·sinθ + z·cosθ
    d_tx = x * tf.sin(angles[:, None, None]) + z * tf.cos(angles[:, None, None]) # (Na, Nz, Nx)
    return d_tx

@tf.function()
def rx_distance(grid, probe_geometry):
    # Vectorised Euclidean distance: √((x−xₑ)² + (y−yₑ)² + (z−zₑ)²)
    d_rx = tf.norm(grid[None, :, :, :] - probe_geometry[:, None, None, :], axis=-1) # (Nc, Nz, Nx)
    return d_rx

@tf.function
def gather_samples(rf, samples_idx):
    # Integer neighbours around each fractional delay
    d0 = tf.cast(tf.floor(samples_idx), "int32")
    d1 = d0 + 1

    rf_0 = tf.gather(rf, d0, axis=1, batch_dims=1)
    rf_1 = tf.gather(rf, d1, axis=1, batch_dims=1)

    # Linear interpolation weights
    d0f = tf.cast(d0, "float32")
    d1f = tf.cast(d1, "float32")
    w0 = d1f - samples_idx
    w1 = samples_idx - d0f

    # Compute linear interpolation
    rf_foc = w0 * rf_0 + w1 * rf_1

    return rf_foc
