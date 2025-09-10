import tensorflow as tf
import numpy as np
from scipy.signal import firwin

from modules.reconstruction.delays import tx_distance, rx_distance, gather_samples
from modules.reconstruction.apod import dynamic_receive_aperture_apod, get_bool_apodization

class ReconstructionRF(tf.Module):

    def __init__(self, pw, grid, device="/gpu:0"):
        super().__init__()
        self.device = device

        with tf.device(self.device):

            self.rfdata = tf.constant(pw.rfdata, dtype=tf.float32)
            self.Nc = tf.constant(pw.n_channels, dtype=tf.int32)
            self.Ns = tf.constant(pw.n_samples, dtype=tf.int32)
            self.Na = tf.constant(pw.n_angles, dtype=tf.int32)
            self.Na_mid = tf.constant(pw.n_angles//2, dtype=tf.int32)
            self.angles = tf.constant(pw.angles, dtype=tf.float32)

            self.fs = tf.constant(pw.fs, dtype=tf.float32)
            self.t0 = tf.constant(pw.t0, dtype=tf.float32)
            self.c0 = tf.constant(pw.c0, dtype=tf.float32)
            self.fc = tf.constant(pw.fc, dtype=tf.float32)
            self.fdemod = tf.constant(pw.fdemod, dtype=tf.float32)
            self.probe_geometry = tf.constant(pw.probe_geometry, dtype=tf.float32)
            self.samples = 0

            self.grid = tf.constant(grid, dtype=tf.float32)

            self.Nz, self.Nx = grid.shape[:-1]
            self.Nz = tf.constant(self.Nz, dtype=tf.int32)
            self.Nx = tf.constant(self.Nx, dtype=tf.int32)

            self.d_tx = tx_distance(self.grid, self.angles)
            self.d_rx = rx_distance(self.grid, self.probe_geometry)
            self.d_tx = tf.constant(self.d_tx, dtype=tf.float32)
            self.d_rx = tf.constant(self.d_rx, dtype=tf.float32)

    def classic_das(self, selected_angles, f_num=1, window="boxcar"):
        with tf.device(self.device):
            self.selected_angles = selected_angles
            self.xlims = (self.probe_geometry[0, 0], self.probe_geometry[-1, 0])
            
            self.rx_apod = dynamic_receive_aperture_apod(self.grid, self.probe_geometry, f_num=f_num, window=window)
            self.rx_apod = tf.constant(self.rx_apod, dtype=tf.float32)

        def _classic_das(angle):
            rf = tf.constant(self.rfdata[angle], dtype=tf.float32)
            delays = self.t0[angle] + (self.d_rx + self.d_tx[angle]) / self.c0
            samples_idx = tf.clip_by_value(self.fs * delays, clip_value_min=0.0, clip_value_max=float(self.Ns))

            # apods = self.rx_apod
            apods = 1
            y_das = tf.reduce_sum(gather_samples(rf, samples_idx) * apods, axis=0, keepdims=False)

            return y_das
        
        y_das =  tf.map_fn(
            fn=_classic_das,
            elems=self.selected_angles,
            fn_output_signature=tf.TensorSpec(shape=(self.Nz, self.Nx), dtype=tf.float32),
            parallel_iterations=1
        )

        return y_das.numpy()

    def dmas(self, selected_angles, f_num=1, window="boxcar", cutoff=np.array([1.5, 2.5])):
        with tf.device(self.device):
            self.selected_angles = selected_angles
            self.xlims = (self.probe_geometry[0, 0], self.probe_geometry[-1, 0])

            self.rx_apod = dynamic_receive_aperture_apod(self.grid, self.probe_geometry, f_num=f_num, window=window)
            self.rx_apod = tf.constant(self.rx_apod, dtype=tf.float32)

        def _dmas(angle):
            rf = tf.constant(self.rfdata[angle], dtype=tf.float32)
            delays = self.t0[angle] + (self.d_rx + self.d_tx[angle]) / self.c0
            samples_idx = tf.clip_by_value(self.fs * delays, clip_value_min=0.0, clip_value_max=float(self.Nz - 2))
            
            s = gather_samples(rf, samples_idx)
            si = tf.sign(s)*tf.sqrt(tf.abs(s))

            apods = self.rx_apod

            ai = tf.square(tf.reduce_sum(si * apods, axis=0, keepdims=False))
            aj = tf.reduce_sum(tf.abs(si * apods), axis=0, keepdims=False)
            y_dmas = 0.5 * (ai - aj)
            
            taps = 128
            fs = self.fs.numpy().item()
            fc = self.fc.numpy().item()
            kernel = firwin(numtaps=taps, cutoff=cutoff * fc, fs=fs, pass_zero=False, window="hamming").astype(np.float32)
            kernel = tf.reshape(kernel[::-1], (taps, 1, 1))

            y_dmas = tf.transpose(y_dmas, [1, 0])
            y_dmas = y_dmas[..., tf.newaxis]

            y_dmas = tf.nn.conv1d(y_dmas, kernel, stride=1, padding="SAME")
            y_dmas = tf.transpose(tf.squeeze(y_dmas, axis=2), [1, 0])

            return y_dmas
        
        y_dmas =  tf.map_fn(
            fn=_dmas,
            elems=self.selected_angles,
            fn_output_signature=tf.TensorSpec(shape=(self.Nz, self.Nx), dtype=tf.float32),
            parallel_iterations=1
        )

        return y_dmas.numpy()

    def mv_bmfrm(self, selected_angles, L=None, delta=None, z_chunk=500):
        with tf.device(self.device):
            self.selected_angles = selected_angles
            _, Nc, _ = self.rfdata.shape 
            Nz, _ = self.grid.shape[:-1]

            L = L or Nc // 2
            # L = 64
            delta = delta or (1.0 / L)
            S = Nc - L + 1

            L = tf.constant(L)
            delta = tf.constant(delta)
            S = tf.constant(S)
            z_chunk = tf.constant(z_chunk)

        def _mv_bmfrm(angle):
            rf = tf.constant(self.rfdata[angle], dtype=tf.float32)
            delays = self.t0[angle] + (self.d_rx + self.d_tx[angle]) / self.c0
            samples_idx = tf.clip_by_value(self.fs * delays, clip_value_min=0.0, clip_value_max=float(self.Ns))

            X = gather_samples(rf, samples_idx)

            a = tf.ones([L], dtype=tf.float32)
            eyeL = tf.eye(L, dtype=tf.float32)

            y_mv = []

            for z0 in range(0, Nz, z_chunk):
                z1 = min(z0 + z_chunk, Nz)
                Xb = X[:, z0:z1, :]
                Zb = z1 - z0

                Xb_t = tf.transpose(Xb, perm=[1,2,0])
                subs = tf.signal.frame(Xb_t, frame_length=L, frame_step=1, axis=-1)

                R = tf.einsum("znsl,znsm->znlm", subs, subs) / tf.cast(S, tf.float32)

                tr = tf.linalg.trace(R) + 1e-3
                R_DL = R + eyeL[None, None, :, :] * (delta * tr)[..., None, None]

                Zb, Nx, _, _ = tf.shape(R_DL)
                rhs0 = tf.reshape(a, [1, 1, L, 1])
                rhs = tf.tile(rhs0, [Zb, Nx, 1, 1])
                u = tf.linalg.solve(R_DL, rhs)
                u = tf.squeeze(u, axis=-1)

                d = tf.einsum("znl,l->zn", u, a)
                w = u / d[..., None]

                y = tf.einsum("znl,znsl->zns", w, subs)
                y_chunk = tf.reduce_mean(y, axis=2)

                y_mv.append(tf.cast(y_chunk, tf.float32))

            y_mv = tf.concat(y_mv, axis=0)

            return y_mv
        
        y_mv =  tf.map_fn(
            fn=_mv_bmfrm,
            elems=self.selected_angles,
            fn_output_signature=tf.TensorSpec(shape=(self.Nz, self.Nx), dtype=tf.float32),
            parallel_iterations=1
        )

        # delays = self.t0[self.Na_mid] + (self.d_rx + self.d_tx[self.Na_mid]) / self.c0
        # samples = tf.clip_by_value(self.fs * delays, clip_value_min=0.0, clip_value_max=float(self.Ns-1))

        # return y_mv.numpy(), tf.cast(samples, tf.uint16).numpy()
        return y_mv.numpy()
    
    def sp_das(self, selected_angles, f_num):

        with tf.device(self.device):

            self.rfdata = tf.reshape(self.rfdata, (self.Na, self.Ns, self.Nc))

            self.selected_angles = selected_angles

            self.mask = get_bool_apodization(self.grid, self.probe_geometry, f_num)
            self.mask = tf.constant(self.mask)
            self.nnz = tf.math.count_nonzero(self.mask) * 2

            self.nnz_per_col = tf.math.count_nonzero(tf.reshape(self.mask, (self.Nc, -1)), axis=1)

            self.mask_idx = tf.constant(tf.cast(tf.where(self.mask), dtype=tf.int32))

            self.rows = self.mask_idx[:, 1] * self.Nx + self.mask_idx[:, 2]
            self.rows = tf.repeat(self.rows, 2)
            self.rows = tf.constant(tf.cast(self.rows, dtype=tf.int64))

        def _sp_das(angle):
            delays = self.t0[angle] + (self.d_rx + self.d_tx[angle]) / self.c0
            samples_idx = tf.clip_by_value(self.fs * delays, clip_value_min=0.0, clip_value_max=float(self.Ns - 2))
            samples_bot = tf.math.floor(samples_idx)
            delta = samples_idx - samples_bot

            samples_bot = tf.cast(samples_bot, tf.int32)
            samples_bot = tf.gather_nd(samples_bot, self.mask_idx)
            
            col = samples_bot + self.Ns * tf.cast(self.mask_idx[:, 0], dtype=tf.int32)
            col = tf.cast(col, tf.int64)
            cols = tf.reshape(tf.stack([col, col+1], axis=1), [-1])

            delta = tf.gather_nd(delta, self.mask_idx)
            vals = tf.reshape(tf.stack([1-delta, delta], axis=1), [-1])

            sp = tf.sparse.SparseTensor(
                indices=tf.stack([self.rows, cols], axis=1),

                values=vals,
                dense_shape=[self.Nz * self.Nx, self.Ns * self.Nc]
            )
            sp = tf.sparse.reorder(sp)

            y_das = tf.sparse.sparse_dense_matmul(sp, tf.reshape(self.rfdata[angle, ...], (self.Ns * self.Nc, 1)))
            y_das = tf.reshape(y_das, (self.Nz, self.Nx))

            return y_das

        y_das = tf.map_fn(
            fn=_sp_das,
            elems=self.selected_angles,
            fn_output_signature=tf.TensorSpec(shape=(self.Nz, self.Nx), dtype=tf.float32),

            parallel_iterations=1
        )

        return y_das.numpy()


    def get_samples(self):
        delays = self.t0[self.Na_mid] + (self.d_rx + self.d_tx[self.Na_mid]) / self.c0
        samples = tf.clip_by_value(self.fs * delays, clip_value_min=0.0, clip_value_max=float(self.Ns))
        samples = tf.cast(samples, tf.uint16)

        return samples.numpy()

        
