import numpy as np
import tensorflow as tf

@tf.function()
def get_bool_apodization(grid, probe_geometry, f_num):
    z = grid[..., 2]
    aperture = z / (2*f_num)
    diff = tf.abs(grid[None, :, :, 0] - probe_geometry[:, None, None, 0])
    mask = (diff <= aperture) # (Nc, Nz, Nx)

    return mask

def get_window(distance, aperture, kind="boxcar"):
    d  = np.asarray(distance)
    a  = np.asarray(np.broadcast_to(aperture, distance.shape))
    w  = np.zeros_like(d, dtype=np.float32)

    if kind == "boxcar":
        w = (d <= a).astype(np.float32)

    elif kind == "hanning":
        mask = d <= a
        w[mask] = 0.5 + 0.5*np.cos(np.pi*d[mask]/a[mask])

    elif kind == "hamming":
        mask = d <= a
        w[mask] = 0.53836 + 0.46164*np.cos(np.pi*d[mask]/a[mask])

    elif kind.startswith("tukey"):
        roll = float(kind[-2:]) / 100.0   # 0.25, 0.50, 0.75
        # Tres zonas: plana, transición cosenoidal, cero
        flat  = d <  a*(1-roll)
        taper = (d >= a*(1-roll)) & (d <= a)
        w[flat]  = 1.0
        w[taper] = 0.5*(1 + np.cos(np.pi/roll*(d[taper]/a[taper] - 1 + roll)))
    else:
        raise ValueError("Ventana no reconocida: "+kind)

    return w

def dynamic_receive_aperture_apod(grid, probe_geometry, f_num=1, window="boxcar"):
    z = grid[..., 2]
    diff = tf.abs(grid[None, :, :, :] - probe_geometry[:, None, None, :]) # (Nc, Nz, Nx, 3)

    aperture = z / (2*f_num)
    lateral_dist = diff[..., 0]

    mask = (lateral_dist <= aperture) # (Nc, Nz, Nx)

    w = get_window(lateral_dist, aperture, kind=window)
    apod = np.where(mask, w, 0.0).astype(np.float32) # (Nc, Nz*Nx)

    return apod
