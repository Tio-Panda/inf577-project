import numpy as np

def get_grid(pw, Nz, Nx):
    z = np.linspace(pw.zlims[0], pw.zlims[-1], Nz)
    x = np.linspace(-pw.aperture_width/2, pw.aperture_width/2, Nx)

    Z, X = np.meshgrid(z, x, indexing="ij")
    Y = X * 0
    grid = np.stack((X, Y, Z), axis=-1, dtype=np.float32)

    return grid

def get_full_grid(pw, zlims, Nz, Nx):
    z = np.linspace(5e-3, pw.img_depth, Nz)
    x = np.linspace(-pw.aperture_width/2, pw.aperture_width/2, Nx)

    Z, X = np.meshgrid(z, x, indexing="ij")
    Y = X * 0
    grid = np.stack((X, Y, Z), axis=-1, dtype=np.float32)

    return grid

def get_custom_grid(pw, zlims, Nz, Nx):
    z = np.linspace(zlims[0], zlims[1], Nz)
    x = np.linspace(-pw.aperture_width/2, pw.aperture_width/2, Nx)

    Z, X = np.meshgrid(z, x, indexing="ij")
    Y = X * 0
    grid = np.stack((X, Y, Z), axis=-1, dtype=np.float32)

    return grid
