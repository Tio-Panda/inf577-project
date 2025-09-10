import tensorflow as tf
import tensorflow_probability as tfp

import keras

class DASBeamformLayer(keras.layers.Layer):
    def __init__(self, Nc, Ns, Nz, Nx, K, **kwargs):
        super().__init__(**kwargs)
        self.Nc = Nc
        self.Ns = Ns
        self.Nz = Nz
        self.Nx = Nx
        self.K = K
        # self._k = _k

    tf.function(jit_compile=True)
    def call(self, inputs):
        def _per_batch(args):
            rf, g, pr, p = args
            c0, fs, t0, _ = tf.unstack(p, axis=-1)

            d_tx = g[..., 2]
            d_rx = tf.norm(g[None, ...] - pr[:, None, None, :], axis=-1)

            samples = fs * (t0 + (d_rx + d_tx[None, ...])) / c0
            samples = tf.reshape(samples, [samples.shape[0], -1])

            def _per_filter(rf_k):
                rf_k = tf.reshape(rf_k, [rf_k.shape[0], -1])

                interp = tfp.math.batch_interp_regular_1d_grid(
                    x=samples, x_ref_min=0.0, x_ref_max=self.Ns - 1, y_ref=rf_k, axis=1
                )

                interp = tf.reshape(interp, shape=(self.Nc, self.Nz, self.Nx))
                y_das = tf.reduce_sum(interp, axis=0, keepdims=False)

                return y_das

            y_das = tf.map_fn(
                _per_filter,
                elems=tf.transpose(rf, perm=[2, 0, 1]),
                parallel_iterations=1,
                swap_memory=True,
                fn_output_signature=tf.TensorSpec(shape=(self.Nz, self.Nx), dtype=tf.float32),
            )

            return y_das

        y_das = tf.map_fn(
            _per_batch,
            elems=inputs,
            parallel_iterations=1,
            swap_memory=False,
            fn_output_signature=tf.TensorSpec(shape=(self.K, self.Nz, self.Nx), dtype=tf.float32),
        )

        y_das = tf.transpose(y_das, perm=[0, 2, 3, 1])
        y_das = tf.ensure_shape(y_das, shape=(None, self.Nz, self.Nx, self.K))
        return y_das

    def compute_output_shape(self):
        return (None, self.Nz, self.Nx, self.K)