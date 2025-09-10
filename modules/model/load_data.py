import os
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf

def get_all_names(main_path):
    p = Path(main_path)
    return {f.stem for f in p.rglob("*.hdf5") if f.is_file()}

def load_data(name, path):
    final_path = path+name+".hdf5"
    with h5py.File(final_path, "r") as f:
        rf = np.array(f["rfdata"]).astype(np.float32)
        grid = np.array(f["grid"]).astype(np.float32)
        probe_geometry = np.array(f["probe_geometry"]).astype(np.float32)
        img = np.array(f["img"]).astype(np.float32)

        rf_max = rf / np.max(np.abs(rf))
        sigma = np.std(rf_max) + 1e-4
        rf = rf_max / sigma

        c0 = np.squeeze(f.attrs["c0"])
        fs = np.squeeze(f.attrs["fs"])
        t0 = np.squeeze(f.attrs["t0"])
        angle = np.squeeze(f.attrs["angle"])

        rf = rf[..., np.newaxis]

        params = np.array([c0, fs, t0, angle]).astype(np.float32)

        inputs = {
            "rf": rf[None, ...], 
            "grid": grid[None, ...], 
            "probe": probe_geometry[None, ...], 
            "params": params[None, ...]}

        label = img

        return inputs, label


class RFDataset(tf.data.Dataset):
    def _generator(filename):
        with h5py.File(filename, "r") as f:
            yield f["rfdata"], f["grid"], f["probe_geometry"], (f.attrs["c0"], f.attrs["fs"], np.squeeze(f.attrs["t0"]), f.attrs["angle"]), f["img"]

    def _preprocessing(rf, grid, probe, params, img):
        rf_max = rf / tf.reduce_max(tf.abs(rf)) + 1e-9
        rf = rf_max / tf.math.reduce_std(rf_max, axis=None, keepdims=False)

        return {
            "rf": rf[..., tf.newaxis],
            "grid": grid,
            "probe": probe,
            "params": params,
        }, img[..., tf.newaxis]


    def __new__(cls, filenames, batch_size=5, buffer_size=5, seed=11, Nc=128, Ns=2800, Nz=2048, Nx=256):

        specs = (
            tf.TensorSpec(shape=(Nc, Ns), dtype=tf.float32, name="rf"),
            tf.TensorSpec(shape=(Nz, Nx, 3), dtype=tf.float32, name="grid"),
            tf.TensorSpec(shape=(Nc, 3), dtype=tf.float32, name="probe"),
            tf.TensorSpec(shape=(4, ), dtype=tf.float32, name="params"),
            tf.TensorSpec(shape=(Nz, Nx), dtype=tf.float32, name="img"),
        )

        n = len(filenames)

        return (tf.data.Dataset.from_tensor_slices(filenames)
            .interleave(lambda filename: tf.data.Dataset.from_generator(
                cls._generator,
                output_signature=specs,
                args=(filename, )
            )
            .map(cls._preprocessing, tf.data.AUTOTUNE).cache()
            .shuffle(buffer_size=buffer_size, seed=seed)
        )
        .apply(tf.data.experimental.assert_cardinality(n))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

def split_dataset(dir_path, splits=(0.85, 0.15), batch_size=5, buffer_size=5, seed=11, Nc=128, Ns=2800, Nz=2048, Nx=256):
    filenames = tf.io.gfile.glob(os.path.join(dir_path, "*.hdf5"))

    n = int(len(filenames) * splits[0])

    ds_train = RFDataset(filenames[:n], batch_size, buffer_size, seed, Nc, Ns, Nz, Nx)
    ds_val = RFDataset(filenames[n:], batch_size, buffer_size, seed, Nc, Ns, Nz, Nx)

    return ds_train, ds_val
