# просторові фільтри: аналоги matlab imfilter та fspecial

import numpy as np
from scipy import ndimage, signal

FILTER_MODES = ("corr", "conv")
BOUNDARY_MODES = ("zero", "replicate", "symmetric", "circular")
SIZE_MODES = ("same", "full")

# назви граничних умов matlab і режими scipy.ndimage
NDIMAGE_BOUNDARY = {
    "zero": ("constant", 0.0),
    "replicate": ("nearest", 0.0),
    "symmetric": ("reflect", 0.0),
    "circular": ("wrap", 0.0),
}

# назви граничних умов matlab і режими numpy.pad
NUMPY_PAD_MODE = {
    "zero": "constant",
    "replicate": "edge",
    "symmetric": "symmetric",
    "circular": "wrap",
}


def _as_kernel(kernel):
    # ядро має бути непорожньою двовимірною матрицею
    array = np.asarray(kernel, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Ядро фільтра має бути непорожньою двовимірною матрицею.")
    return array


def _as_image(image):
    # зображення теж має бути двовимірним
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Зображення має бути непорожньою двовимірною матрицею.")
    return array


def _boundary_spec(boundary):
    # який режим scipy.ndimage відповідає граничній умові
    if isinstance(boundary, str):
        key = boundary.lower()
        if key not in NDIMAGE_BOUNDARY:
            raise ValueError(
                f"Невідома гранична умова {boundary!r}; "
                f"допустимі: {', '.join(BOUNDARY_MODES)}."
            )
        return NDIMAGE_BOUNDARY[key]
    if np.isscalar(boundary):
        return "constant", float(boundary)
    raise ValueError("Гранична умова має бути назвою або числом.")


def _pad_image(image, kernel, boundary):
    # розширюємо зображення на m-1 рядків і n-1 стовпців з кожного боку
    h, w = kernel.shape
    pad_width = ((h - 1, h - 1), (w - 1, w - 1))

    if isinstance(boundary, str):
        key = boundary.lower()
        if key not in NUMPY_PAD_MODE:
            raise ValueError(
                f"Невідома гранична умова {boundary!r}; "
                f"допустимі: {', '.join(BOUNDARY_MODES)}."
            )
        if key == "zero":
            return np.pad(image, pad_width, mode="constant", constant_values=0.0)
        return np.pad(image, pad_width, mode=NUMPY_PAD_MODE[key])

    if np.isscalar(boundary):
        return np.pad(
            image, pad_width, mode="constant", constant_values=float(boundary)
        )
    raise ValueError("Гранична умова має бути назвою або числом.")


def imfilter(image, kernel, mode="corr", boundary="zero", size="same"):
    # фільтруємо зображення ядром, як це робить matlab imfilter
    image_array = _as_image(image)
    kernel_array = _as_kernel(kernel)

    mode_key = mode.lower()
    if mode_key not in FILTER_MODES:
        raise ValueError(f"Невідомий режим {mode!r}; допустимі: corr, conv.")

    size_key = size.lower()
    if size_key not in SIZE_MODES:
        raise ValueError(f"Невідомий розмір {size!r}; допустимі: same, full.")

    # для same scipy сам розширює зображення і бере центр
    if size_key == "same":
        nd_mode, cval = _boundary_spec(boundary)
        function = ndimage.correlate if mode_key == "corr" else ndimage.convolve
        if nd_mode == "constant":
            return function(image_array, kernel_array, mode=nd_mode, cval=cval)
        return function(image_array, kernel_array, mode=nd_mode)

    # для full доповнюємо зображення і рахуємо в режимі valid
    padded = _pad_image(image_array, kernel_array, boundary)
    function = signal.correlate2d if mode_key == "corr" else signal.convolve2d
    return function(padded, kernel_array, mode="valid")


def _size_pair(size):
    # розмір ядра: або одне число, або пара (рядки, стовпці)
    if np.isscalar(size):
        h = w = int(size)
    else:
        values = tuple(int(v) for v in size)
        if len(values) != 2:
            raise ValueError("Розмір ядра має бути скаляром або парою (рядки, стовпці).")
        h, w = values
    if h <= 0 or w <= 0:
        raise ValueError("Розміри ядра мають бути додатними.")
    return h, w


def average_kernel(size=3):
    # усереднювальний фільтр, сума ядра 1
    h, w = _size_pair(size)
    return np.full((h, w), 1.0 / (h * w), dtype=np.float64)


def disk_kernel(radius=5):
    # кругове усереднення, сума ядра 1
    r = int(radius)
    if r <= 0:
        raise ValueError("Радіус диска має бути додатним.")
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    mask = (xx * xx + yy * yy <= r * r).astype(np.float64)
    return mask / mask.sum()


def gaussian_kernel(size=3, sigma=0.5):
    # гауссів фільтр, сума ядра 1
    h, w = _size_pair(size)
    if sigma <= 0:
        raise ValueError("Sigma має бути додатною.")
    y = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
    x = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def laplacian_kernel(alpha=0.2):
    # лапласіан, сума ядра 0
    a = float(np.clip(alpha, 0.0, 1.0))
    denominator = a + 1.0
    return np.array(
        [
            [a / denominator, (1.0 - a) / denominator, a / denominator],
            [
                (1.0 - a) / denominator,
                -4.0 / denominator,
                (1.0 - a) / denominator,
            ],
            [a / denominator, (1.0 - a) / denominator, a / denominator],
        ]
    )


def log_kernel(size=5, sigma=0.5):
    # лапласіан-гаус, сума ядра 0
    h, w = _size_pair(size)
    if sigma <= 0:
        raise ValueError("Sigma має бути додатною.")
    y = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
    x = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radius_sq = xx * xx + yy * yy
    envelope = np.exp(-radius_sq / (2.0 * sigma * sigma))
    envelope[envelope < np.finfo(np.float64).eps * envelope.max()] = 0.0
    envelope /= envelope.sum()
    kernel = envelope * (radius_sq - 2.0 * sigma * sigma) / (sigma**4)
    kernel -= kernel.mean()
    return kernel


def sobel_kernel():
    # собель, сума ядра 0
    return np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])


def prewitt_kernel():
    # превитт, сума ядра 0
    return np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])


def unsharp_kernel(alpha=0.2):
    # нерізке маскування, сума ядра 1
    a = float(np.clip(alpha, 0.0, 1.0))
    kernel = np.array(
        [
            [-a, a - 1.0, -a],
            [a - 1.0, a + 5.0, a - 1.0],
            [-a, a - 1.0, -a],
        ]
    )
    return kernel / (a + 1.0)


def fspecial(kind, *args, **kwargs):
    # повертаємо ядро за назвою, як у matlab fspecial
    name = kind.lower()
    if name == "average":
        size = args[0] if args else kwargs.get("size", kwargs.get("hsize", 3))
        return average_kernel(size)
    if name == "disk":
        radius = args[0] if args else kwargs.get("radius", 5)
        return disk_kernel(radius)
    if name == "gaussian":
        size = args[0] if args else kwargs.get("size", kwargs.get("hsize", 3))
        sigma = args[1] if len(args) > 1 else kwargs.get("sigma", 0.5)
        return gaussian_kernel(size, sigma)
    if name == "laplacian":
        alpha = args[0] if args else kwargs.get("alpha", 0.2)
        return laplacian_kernel(alpha)
    if name == "log":
        size = args[0] if args else kwargs.get("size", kwargs.get("hsize", 5))
        sigma = args[1] if len(args) > 1 else kwargs.get("sigma", 0.5)
        return log_kernel(size, sigma)
    if name == "sobel":
        return sobel_kernel()
    if name == "prewitt":
        return prewitt_kernel()
    if name == "unsharp":
        alpha = args[0] if args else kwargs.get("alpha", 0.2)
        return unsharp_kernel(alpha)
    raise ValueError(
        f"Невідомий тип фільтра {kind!r}; допустимі: average, disk, gaussian, "
        "laplacian, log, sobel, prewitt, unsharp."
    )


def laplacian5_kernel():
    # класичне ядро лапласа 5 на 5
    kernel = np.zeros((5, 5), dtype=np.float64)
    kernel[2, 2] = 16.0
    kernel[1, 2] = kernel[3, 2] = kernel[2, 1] = kernel[2, 3] = -2.0
    kernel[0, 2] = kernel[4, 2] = kernel[2, 0] = kernel[2, 4] = -1.0
    kernel[1, 1] = kernel[1, 3] = kernel[3, 1] = kernel[3, 3] = -1.0
    return kernel


def sharpening_kernel(strength=0.5):
    # власне ядро підвищення різкості: дельта плюс strength * лапласіан
    delta = np.zeros((5, 5), dtype=np.float64)
    delta[2, 2] = 1.0
    return delta + float(strength) * laplacian5_kernel()


def kernel_metadata(kernel):
    # короткі відомості про ядро для results.json
    array = np.asarray(kernel, dtype=np.float64)
    return {
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "sum": float(array.sum()),
        "min": float(array.min()),
        "max": float(array.max()),
        "nonzero": int(np.count_nonzero(array)),
    }
