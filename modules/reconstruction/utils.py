import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from PIL import Image


def get_bmode(img):
    env = hilbert(img, axis=0)
    bimg = 20 * np.log10(np.abs(env) + 1e-10)
    bimg -= np.amax(bimg)

    return bimg


def show_bmode(bimg, grid, vmin=-60):
    zlims = grid[:, 0, 2] * 1e3
    xlims = grid[0, :, 0] * 1e3

    plt.figure(figsize=(5, 8))
    plt.imshow(
        bimg,
        vmin=vmin,
        cmap="gray",
        extent=(xlims[0], xlims[-1], zlims[-1], zlims[0]),
        origin="upper",
    )

    plt.xlabel("Lateral [mm]")
    plt.ylabel("Axial [mm]")
    plt.show()

def save_bmode(bimg, grid, path, vmin=-60, vmax=0):
    bimg = np.clip(bimg, vmin, vmax)
    bimg = (bimg - vmin) / (vmax - vmin)
    img = (bimg * 255).astype(np.uint8)

    zlims = grid[:, 0, 2] * 1e3
    xlims = grid[0, :, 0] * 1e3
    aspect = (xlims[-1] - xlims[0]) / (zlims[-1] - zlims[0])

    Nz, _ = img.shape
    new_w = max(1, int(round(Nz * aspect)))
    
    img = Image.fromarray(img, mode="L")
    img = img.resize((new_w, Nz), resample=Image.BILINEAR)
    img.save(path)