# функції для спектра і синтезу фільтрів, аналоги matlab freqspace/fsamp2/fwind

import numpy as np
from scipy import ndimage


def to_uint8(array):
    # обрізаємо значення до 0...255
    return np.clip(
        np.rint(np.asarray(array, dtype=np.float64)), 0.0, 255.0
    ).astype(np.uint8)


def fft2(image, shape=None):
    # двовимірне дпф, за потреби доповнюємо нулями до потрібного розміру
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("fft2 очікує двовимірний масив.")
    if shape is not None:
        rows, columns = int(shape[0]), int(shape[1])
        if rows < data.shape[0] or columns < data.shape[1]:
            raise ValueError("Розмір доповнення не може бути меншим за вхід.")
        padded = np.zeros((rows, columns), dtype=np.float64)
        padded[: data.shape[0], : data.shape[1]] = data
        data = padded
    return np.fft.fft2(data)


def abs_spectrum(spectrum):
    # модуль комплексного спектра
    return np.abs(np.asarray(spectrum))


def fftshift(spectrum):
    # переносимо нульову частоту в центр масиву
    return np.fft.fftshift(np.asarray(spectrum))


def ifftshift(spectrum):
    # обернена операція до fftshift
    return np.fft.ifftshift(np.asarray(spectrum))


def _frequency_axis(n):
    # вісь нормованих частот, для парного n від -1 до 1-2/n
    n = int(n)
    if n <= 0:
        raise ValueError("Кількість відліків частоти має бути додатною.")
    step = 2.0 / n
    start = -1.0 if n % 2 == 0 else -1.0 + 1.0 / n
    return start + np.arange(n, dtype=np.float64) * step


def freqspace_1d(n, whole=False):
    # freqspace(n) або freqspace(n, 'whole')
    n = int(n)
    if n <= 0:
        raise ValueError("Кількість точок має бути додатною.")
    if whole:
        return np.arange(n, dtype=np.float64) * (2.0 / n)
    return np.arange(n // 2 + 1, dtype=np.float64) * (2.0 / n)


def freqspace(shape, meshgrid=False):
    # сітка нормованих частот, повертає осі (fy, fx)
    if isinstance(shape, (int, np.integer)):
        rows = columns = int(shape)
    else:
        values = list(shape)
        if len(values) != 2:
            raise ValueError("shape має містити два розміри (рядки, стовпці).")
        rows, columns = int(values[0]), int(values[1])
    fy = _frequency_axis(rows)
    fx = _frequency_axis(columns)
    if meshgrid:
        return np.meshgrid(fx, fy)
    return fy, fx


def paddedsize(shape, kernel_shape=None):
    # розмір доповнення перед фільтрацією
    rows, columns = int(shape[0]), int(shape[1])
    if kernel_shape is None:
        return 2 * rows, 2 * columns
    kernel_rows, kernel_columns = int(kernel_shape[0]), int(kernel_shape[1])
    padded_rows = rows + kernel_rows - 1
    padded_columns = columns + kernel_columns - 1
    if padded_rows % 2:
        padded_rows += 1
    if padded_columns % 2:
        padded_columns += 1
    return padded_rows, padded_columns


def pad_image(image, shape, mode="reflect"):
    # доповнюємо зображення праворуч і знизу
    data = np.asarray(image, dtype=np.float64)
    rows, columns = int(shape[0]), int(shape[1])
    if rows < data.shape[0] or columns < data.shape[1]:
        raise ValueError("Розмір доповнення не може бути меншим за вхід.")
    pad_rows = rows - data.shape[0]
    pad_columns = columns - data.shape[1]
    if mode == "reflect":
        return np.pad(data, ((0, pad_rows), (0, pad_columns)), mode="reflect")
    if mode == "zero":
        padded = np.zeros((rows, columns), dtype=np.float64)
        padded[: data.shape[0], : data.shape[1]] = data
        return padded
    if mode == "edge":
        return np.pad(data, ((0, pad_rows), (0, pad_columns)), mode="edge")
    raise ValueError(f"Невідомий режим доповнення: {mode}")


def fsamp2(desired):
    # синтез ядра методом частотної вибірки
    response = np.asarray(desired, dtype=np.float64)
    if response.ndim != 2:
        raise ValueError("Бажана частотна характеристика має бути двовимірною.")
    if not np.all(np.isfinite(response)):
        raise ValueError("Бажана характеристика містить NaN або нескінченність.")
    kernel = np.real(np.fft.ifft2(np.fft.ifftshift(response)))
    return np.fft.fftshift(kernel)


def hamming_window(n):
    # одновимірне вікно хеммінга
    return np.hamming(int(n))


def radial_hann_window(n):
    # двовимірне радіальне вікно ханна, не добуток одновимірних вікон
    n = int(n)
    if n <= 0:
        raise ValueError("Розмір вікна має бути додатним.")
    yy, xx = np.mgrid[0:n, 0:n]
    center = (n - 1) / 2.0
    radius = np.hypot(yy - center, xx - center)
    maximum = center if center > 0 else 1.0
    window = 0.5 * (1.0 + np.cos(np.pi * np.minimum(radius, maximum) / maximum))
    window[radius > maximum] = 0.0
    return window


def fwind1(desired, window1, window2=None, normalize_dc=True):
    # синтез ядра віконним методом з одновимірних вікон
    response = np.asarray(desired, dtype=np.float64)
    first = np.asarray(window1, dtype=np.float64).ravel()
    second = first if window2 is None else np.asarray(window2, dtype=np.float64).ravel()
    if response.ndim != 2:
        raise ValueError("Бажана характеристика має бути двовимірною.")
    if first.size != response.shape[0] or second.size != response.shape[1]:
        raise ValueError(
            "Довжини вікон мають збігатися з розмірами бажаної характеристики."
        )
    window2d = np.outer(first, second)
    kernel = fsamp2(response) * window2d
    if normalize_dc:
        total = float(kernel.sum())
        if abs(total) < 1e-12:
            raise ValueError("Сума зваженого ядра близька до нуля.")
        kernel = kernel / total
    return kernel


def fwind2(desired, window2d, normalize_dc=True):
    # синтез ядра з довільним двовимірним вікном
    response = np.asarray(desired, dtype=np.float64)
    window = np.asarray(window2d, dtype=np.float64)
    if response.ndim != 2 or window.shape != response.shape:
        raise ValueError(
            "Вікно має мати такий самий розмір, як бажана характеристика."
        )
    if not np.all(np.isfinite(window)):
        raise ValueError("Вікно містить NaN або нескінченність.")
    kernel = fsamp2(response) * window
    if normalize_dc:
        total = float(kernel.sum())
        if abs(total) < 1e-12:
            raise ValueError("Сума зваженого ядра близька до нуля.")
        kernel = kernel / total
    return kernel


def desired_lowpass(shape, cutoff):
    # ідеальна кругова нч-характеристика і радіус сітки
    rows, columns = int(shape[0]), int(shape[1])
    if not 0.0 < cutoff < 1.0:
        raise ValueError("Зріз має бути в інтервалі (0, 1).")
    fy, fx = freqspace((rows, columns))
    radius = np.hypot(fy[:, None], fx[None, :])
    return (radius <= cutoff).astype(np.float64), radius


def custom_lowpass_response(shape, inner, outer):
    # кругова нч-характеристика з косинусним переходом
    rows, columns = int(shape[0]), int(shape[1])
    if not 0.0 <= inner < outer < 1.0:
        raise ValueError("Потрібно 0 <= inner < outer < 1.")
    fy, fx = freqspace((rows, columns))
    radius = np.hypot(fy[:, None], fx[None, :])
    response = np.zeros((rows, columns), dtype=np.float64)
    response[radius <= inner] = 1.0
    transition = (radius > inner) & (radius < outer)
    response[transition] = 0.5 * (
        1.0 + np.cos(np.pi * (radius[transition] - inner) / (outer - inner))
    )
    return response, radius


def kernel_frequency_response(kernel, shape):
    # ачх ядра на сітці freqspace(shape)
    data = np.asarray(kernel, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Ядро має бути двовимірним.")
    rows, columns = int(shape[0]), int(shape[1])
    fy, fx = freqspace((rows, columns))
    kernel_rows, kernel_columns = data.shape
    offsets_y = np.arange(kernel_rows, dtype=np.float64) - kernel_rows // 2
    offsets_x = np.arange(kernel_columns, dtype=np.float64) - kernel_columns // 2
    basis_y = np.exp(-1j * np.pi * np.outer(fy, offsets_y))
    basis_x = np.exp(-1j * np.pi * np.outer(fx, offsets_x))
    return basis_y @ data @ basis_x.T


def frequency_filter(
    image, response_centered, pad_shape=None, pad_mode="reflect", remove_mean=True
):
    # фільтруємо зображення множенням спектра на передатну функцію
    data = np.asarray(image, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Зображення має бути двовимірним.")
    rows, columns = data.shape
    if pad_shape is None:
        pad_shape = paddedsize(data.shape)
    pad_shape = (int(pad_shape[0]), int(pad_shape[1]))
    response = np.asarray(response_centered)
    if response.shape != pad_shape:
        raise ValueError("Розмір характеристики має дорівнювати розміру доповнення.")
    mean_value = float(data.mean()) if remove_mean else 0.0
    padded = pad_image(data - mean_value, pad_shape, mode=pad_mode)
    spectrum = np.fft.fft2(padded)
    shifted_response = np.fft.fftshift(response)
    filtered = np.real(np.fft.ifft2(spectrum * shifted_response))
    return filtered[:rows, :columns] + mean_value


def filter_with_kernel(image, kernel, pad_shape=None, pad_mode="reflect"):
    # синтез ачх ядра і фільтрація
    if pad_shape is None:
        pad_shape = paddedsize(image.shape)
    response = kernel_frequency_response(kernel, pad_shape).real
    return frequency_filter(
        image, response, pad_shape=pad_shape, pad_mode=pad_mode
    )


def radial_profile(data, bins=64, max_radius=1.0):
    # усереднюємо масив у кільцях частоти, повертає (центри, значення)
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("Масив має бути двовимірним.")
    fy, fx = freqspace(values.shape)
    radius = np.hypot(fy[:, None], fx[None, :])
    edges = np.linspace(0.0, float(max_radius), int(bins) + 1)
    indices = np.digitize(radius.ravel(), edges) - 1
    flat = values.ravel()
    centers = []
    means = []
    for index in range(int(bins)):
        mask = indices == index
        if not np.any(mask):
            continue
        centers.append(float((edges[index] + edges[index + 1]) / 2.0))
        means.append(flat[mask].mean())
    return np.asarray(centers), np.asarray(means)


def power_spectrum(image):
    # спектр потужності після віднімання середнього
    data = np.asarray(image, dtype=np.float64)
    centered = data - data.mean()
    return np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2


def spectral_energy_fraction(image, radius):
    # частка енергії поза заданим радіусом і середня потужність там
    power = power_spectrum(image)
    fy, fx = freqspace(power.shape)
    grid_radius = np.hypot(fy[:, None], fx[None, :])
    mask = grid_radius > float(radius)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0, 0.0
    high = float(power[mask].sum())
    return high / total, float(power[mask].mean())


def highpass_energy(image, sigma=2.0):
    # середній квадрат залишку після гауссового згладжування
    data = np.asarray(image, dtype=np.float64)
    blurred = ndimage.gaussian_filter(data, sigma=float(sigma), mode="reflect")
    residual = data - blurred
    return float(np.mean(residual * residual))


def laplacian_energy(image):
    # середній квадрат відгуку лапласіана, міра різкості
    data = np.asarray(image, dtype=np.float64)
    response = ndimage.laplace(data, mode="reflect")
    return float(np.mean(response * response))


def gradient_magnitude(image):
    # модуль градієнта за оператором собеля
    data = np.asarray(image, dtype=np.float64)
    gradient_x = ndimage.sobel(data, axis=1, mode="nearest")
    gradient_y = ndimage.sobel(data, axis=0, mode="nearest")
    return np.hypot(gradient_x, gradient_y)


def edge_metrics(reference, test):
    # наскільки вціліли границі відносно еталона
    reference_gradient = gradient_magnitude(reference)
    test_gradient = gradient_magnitude(test)
    if reference_gradient.std() < 1e-12 or test_gradient.std() < 1e-12:
        global_correlation = 0.0
    else:
        global_correlation = float(
            np.corrcoef(
                reference_gradient.ravel(), test_gradient.ravel()
            )[0, 1]
        )
    threshold = float(np.percentile(reference_gradient, 90.0))
    strong = reference_gradient >= threshold
    if strong.sum() < 10 or test_gradient[strong].std() < 1e-12:
        strong_correlation = 0.0
    else:
        strong_correlation = float(
            np.corrcoef(reference_gradient[strong], test_gradient[strong])[0, 1]
        )
    reference_energy = float(np.mean(reference_gradient**2))
    test_energy = float(np.mean(test_gradient**2))
    energy_ratio = test_energy / reference_energy if reference_energy > 0 else 0.0
    return {
        "gradient_correlation": global_correlation,
        "strong_edge_correlation": strong_correlation,
        "gradient_energy_ratio": energy_ratio,
    }
