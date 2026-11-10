from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8).copy()


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def save_gray(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_jpeg(path, image, quality):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image, 0, 255).astype(np.uint8)
    mode = "L" if array.ndim == 2 else "RGB"
    Image.fromarray(array, mode=mode).save(
        path, format="JPEG", quality=int(quality), optimize=False
    )


def save_figure(figure, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def image_axis(axis, image, title, cmap="gray", vmin=None, vmax=None):
    axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def histogram_axis(axis, image, title, bins=256, value_range=(0, 255)):
    axis.hist(image.ravel(), bins=bins, range=value_range, color="0.3")
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("рівень яскравості")
    axis.set_ylabel("кількість пікселів")
    axis.grid(alpha=0.2)


def mse(reference, test):
    a = reference.astype(np.float64)
    b = test.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(reference, test, data_range=255.0):
    error = mse(reference, test)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / error))


def ssim(reference, test, data_range=255.0):
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            reference.astype(np.float64),
            test.astype(np.float64),
            data_range=data_range,
        )
    )
