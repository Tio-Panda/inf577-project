import os
import h5py
import numpy as np
import tensorflow as tf

Nc, Ns, Nz, Nx = 128, 2800, 2048, 256
specs = (
    tf.TensorSpec(shape=(Nc, Ns), dtype=tf.float32, name="rf"),
    tf.TensorSpec(shape=(Nc, Nz, Nx), dtype=tf.uint16, name="samples"),
    tf.TensorSpec(shape=(Nz, Nx), dtype=tf.float32, name="img"),
)

class RFPrecalDataset(tf.data.Dataset):
    def _generator(filename):
        with h5py.File(filename, "r") as f:
            yield f["rfdata"], f["samples"], f["img"]

    def _preprocessing(rf, samples, img):
        return {"rf": rf[..., tf.newaxis], "samples": samples}, img[..., tf.newaxis]

    def __new__(cls, filenames, batch_size=5, buffer_size=5, seed=11):
        return (tf.data.Dataset.from_tensor_slices(filenames)
            .interleave(lambda filename: tf.data.Dataset.from_generator(
                cls._generator,
                output_signature=specs,
                args=(filename, )
            )
            .map(cls._preprocessing, tf.data.AUTOTUNE).cache()
            .shuffle(buffer_size=buffer_size, seed=seed)
        )

        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)

    )

def split_precalculated_dataset(dir_path, splits=(0.85, 0.15), batch_size=5, buffer_size=5, seed=11):
    filenames = tf.io.gfile.glob(os.path.join(dir_path, "*.hdf5"))

    n = int(len(filenames) * splits[0])

    ds_train = RFPrecalDataset(filenames[:n], batch_size, buffer_size, seed)
    ds_val = RFPrecalDataset(filenames[n:], batch_size, buffer_size, seed)

    return ds_train, ds_val

def load_precalculated_data(name, path):
    final_path = path+name+".hdf5"
    with h5py.File(final_path, "r") as f:
        rf = np.array(f["rfdata"])
        samples = np.array(f["samples"])
        grid = np.array(f["grid"])

        img = np.array(f["img"])

        rf = rf[..., np.newaxis]

        inputs = (rf[None, ...], samples[None, ...])
        label = img

        return inputs, label, grid
