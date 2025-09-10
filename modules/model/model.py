from keras import Input, Model
import tensorflow as tf
from keras.layers import Conv2D, Lambda, LeakyReLU

from modules.model.layers import DASBeamformLayer

def get_model(
    k_array, version="FULL", Nc=128, Ns=2500, Nz=2024, Nx=256
):
    i_rf = Input(shape=(Nc, Ns, 1), name="rf")
    i_grid = Input(shape=(Nz, Nx, 3), name="grid")
    i_probe = Input(shape=(Nc, 3), name="probe")
    i_params = Input(shape=(4, ), name="params")

    inputs = {"rf": i_rf, "grid": i_grid, "probe": i_probe, "params": i_params}

    if version == "SIMPLE":
        bmfrm = DASBeamformLayer(Nc, Ns, Nz, Nx, 1)([i_rf, i_grid, i_probe, i_params])
        model = Model(inputs=inputs, outputs=bmfrm)
        model.compile(optimizer="adam", loss="mean_squared_error")

        return model

    x = i_rf
    n = len(k_array)
    kernels = [(5, 3), (7, 5)]

    for i, k in enumerate(k_array):
        if k == -1:
            if i == 0 or i == n-1:
                x = DASBeamformLayer(Nc, Ns, Nz, Nx, 1)([x, i_grid, i_probe, i_params])
                continue

            x = DASBeamformLayer(Nc, Ns, Nz, Nx, k_array[i-1])([x, i_grid, i_probe, i_params])
            continue

        kernel = kernels[0] if i != n-1 else kernels[1]
        x = Conv2D(k, kernel, activation=LeakyReLU(0.01), padding="same", name=f"conv{i}")(x)


    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model



# i = 0
# k = n_max_filters
# while k > 0:
#     if idx_bmfrm == i:
#         k *= 2
#         _k = 1 if idx_bmfrm == 0 else k
#         x = DASBeamformLayer(Nc, Ns, Nz, Nx, _k)([x, i_grid, i_probe, i_params])
#         i += 1
#         continue
#     
#     kernel = kernels[0] if k != 1 else kernels[1]
#     x = Conv2D(k, kernel, activation=LeakyReLU(0.01), padding="same", name=f"conv{i}")(x)
#     k //= 2
#     i += 1
#     
#     if idx_bmfrm == i and k == 0:
#         x = DASBeamformLayer(Nc, Ns, Nz, Nx, 1)([x, i_grid, i_probe, i_params])
#         break
