# практична робота 4
# обробка зображень цифровими фільтрами
# запуск: python main.py

import json
import sys
from pathlib import Path

import matplotlib

# щоб не відкривати вікно matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage.restoration import denoise_bilateral

import filters
from utils import (
    histogram_axis,
    image_axis,
    load_gray,
    mse,
    psnr,
    save_figure,
    save_gray,
    ssim,
)

# щоб кирилиця не ламалась в консолі
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
IMAGE_DIR = OUT_DIR / "images"

SEED = 20241104
DISPLAY_MAX = 160.0

# список стандартних ядер
KERNEL_SPECS = [
    {
        "name": "average",
        "label": "Усереднення",
        "matlab": "average",
        "call": "fspecial('average', 5)",
        "args": (5,),
        "params": {"size": [5, 5]},
        "highpass": False,
    },
    {
        "name": "disk",
        "label": "Кругове усереднення",
        "matlab": "disk",
        "call": "fspecial('disk', 5)",
        "args": (5,),
        "params": {"radius": 5},
        "highpass": False,
    },
    {
        "name": "gaussian",
        "label": "Гауссів фільтр",
        "matlab": "gaussian",
        "call": "fspecial('gaussian', 9, 0.8)",
        "args": (9, 0.8),
        "params": {"size": [9, 9], "sigma": 0.8},
        "highpass": False,
    },
    {
        "name": "laplacian",
        "label": "Лапласіан",
        "matlab": "laplacian",
        "call": "fspecial('laplacian', 0.2)",
        "args": (0.2,),
        "params": {"alpha": 0.2},
        "highpass": True,
    },
    {
        "name": "log",
        "label": "Лапласіан-Гаус",
        "matlab": "log",
        "call": "fspecial('log', 5, 0.5)",
        "args": (5, 0.5),
        "params": {"size": [5, 5], "sigma": 0.5},
        "highpass": True,
    },
    {
        "name": "sobel",
        "label": "Собель",
        "matlab": "sobel",
        "call": "fspecial('sobel')",
        "args": (),
        "params": {},
        "highpass": True,
    },
    {
        "name": "prewitt",
        "label": "Превитт",
        "matlab": "prewitt",
        "call": "fspecial('prewitt')",
        "args": (),
        "params": {},
        "highpass": True,
    },
    {
        "name": "unsharp",
        "label": "Нерізке маскування",
        "matlab": "unsharp",
        "call": "fspecial('unsharp', 0.2)",
        "args": (0.2,),
        "params": {"alpha": 0.2},
        "highpass": False,
    },
]


def to_uint8(array):
    # обрізаємо значення в межах 0..255
    return np.clip(np.rint(array), 0.0, 255.0).astype(np.uint8)


def image_stats(array):
    # основні числа зображення
    data = np.asarray(array, dtype=np.float64)
    return {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def rel_path(path):
    # шлях відносно папки практичної роботи
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def save_result_image(name, array):
    # окреме зображення результату в out/images
    path = IMAGE_DIR / f"{name}.png"
    save_gray(path, array)
    return rel_path(path)


def display_window(image):
    # вікно підсилення контрасту для перегляду
    low, high = np.percentile(image.astype(np.float64), (2.0, 98.0))
    if high - low < 10.0:
        low = max(0.0, low - 5.0)
        high = min(255.0, high + 5.0)
    return float(np.floor(low)), float(np.ceil(high))


def response_visual(response, symmetric=False):
    # робимо картинку з відгуку: двополярну або по модулю
    magnitude = np.abs(response.astype(np.float64))
    scale = float(np.percentile(magnitude, 99.5))
    scale = max(scale, 1e-9)
    if symmetric:
        visual = 127.5 + 127.5 * np.clip(response / scale, -1.0, 1.0)
    else:
        visual = 255.0 * np.clip(magnitude / scale, 0.0, 1.0)
    return to_uint8(visual)


def kernel_axis(axis, kernel, title):
    # показуємо ядро як теплову карту
    if float(kernel.min()) < -1e-12:
        limit = float(np.max(np.abs(kernel)))
        image = axis.imshow(
            kernel,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
    else:
        image = axis.imshow(kernel, cmap="viridis", interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def border_mask(shape, width):
    # маска смуги шириною width уздовж усіх країв
    mask = np.zeros(shape, dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def edge_correlation(reference, test):
    # кореляція модулів градієнтів, міра збереження границь
    reference_data = reference.astype(np.float64)
    test_data = test.astype(np.float64)
    reference_gradient = np.hypot(
        ndimage.sobel(reference_data, axis=1, mode="nearest"),
        ndimage.sobel(reference_data, axis=0, mode="nearest"),
    )
    test_gradient = np.hypot(
        ndimage.sobel(test_data, axis=1, mode="nearest"),
        ndimage.sobel(test_data, axis=0, mode="nearest"),
    )
    if reference_gradient.std() < 1e-12 or test_gradient.std() < 1e-12:
        return 0.0
    return float(
        np.corrcoef(reference_gradient.ravel(), test_gradient.ravel())[0, 1]
    )


def metrics_vs(reference, test):
    # метрики відносно еталона
    return {
        "mse": float(mse(reference, test)),
        "psnr": float(psnr(reference, test)),
        "ssim": float(ssim(reference, test)),
        "edge_correlation": edge_correlation(reference, test),
    }


def figure_input(image, vmin, vmax):
    # рисунок 1: вхідне зображення, контраст і гістограма
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    data_max = float(image.max())
    image_axis(
        axes[0],
        image,
        f"а) вхідне зображення\nшкала 0-{data_max:.0f} "
        f"(дані {image.min()}-{data_max:.0f})",
        vmin=0.0,
        vmax=data_max,
    )
    image_axis(
        axes[1],
        image,
        f"б) контраст підсилено\nвікно {vmin:.0f}-{vmax:.0f}",
        vmin=vmin,
        vmax=vmax,
    )
    histogram_axis(
        axes[2],
        image,
        "в) гістограма яскравості",
        bins=128,
        value_range=(0, int(data_max) + 1),
    )
    figure.suptitle(
        "Рисунок 1. Вхідне напівтонове зображення та його гістограма",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig01_input_image.png")


def boundary_experiment(checkerboard, kernel):
    # дивимось як працюють різні граничні умови
    radius = max(1, min(kernel.shape) // 2)
    mask = border_mask(checkerboard.shape, radius)

    figure, axes = plt.subplots(2, 5, figsize=(21, 9))
    image_axis(
        axes[0, 0],
        checkerboard,
        f"Оригінал\n{checkerboard.shape[0]}×{checkerboard.shape[1]}",
        vmin=0.0,
        vmax=255.0,
    )
    image_axis(
        axes[1, 0],
        checkerboard[:96, :96],
        "Оригінал, фрагмент 96×96",
        vmin=0.0,
        vmax=255.0,
    )

    responses = {}
    entries = {}
    for column, boundary in enumerate(filters.BOUNDARY_MODES, start=1):
        response = filters.imfilter(
            checkerboard, kernel, mode="corr", boundary=boundary, size="same"
        )
        result = to_uint8(response)
        responses[boundary] = response
        path = save_result_image(f"checkerboard_{boundary}", result)
        entries[boundary] = {
            "image": path,
            "shape": [int(result.shape[0]), int(result.shape[1])],
            "mean": float(result.mean()),
        }
        image_axis(
            axes[0, column],
            result,
            f"{boundary}\nсереднє = {result.mean():.2f}",
            vmin=0.0,
            vmax=255.0,
        )
        image_axis(
            axes[1, column],
            result[:96, :96],
            f"{boundary}, фрагмент",
            vmin=0.0,
            vmax=255.0,
        )

    # рахуємо відмінності від replicate
    reference = responses["replicate"]
    for boundary, response in responses.items():
        difference = np.abs(response - reference)
        entries[boundary]["border_mae"] = float(difference[mask].mean())
        entries[boundary]["global_mae"] = float(difference.mean())

    figure.suptitle(
        "Рисунок 2. Вплив граничних умов imfilter (усереднення 5×5, кореляція)\n"
        "верхній ряд: усе зображення, нижній: збільшений лівий верхній кут",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig02_imfilter_boundary_modes.png")

    return {
        "image": "data/checkerboard.png",
        "kernel": "average 5x5",
        "filter_mode": "corr",
        "size": "same",
        "modes": entries,
    }


def correlation_convolution_experiment(image, kernel):
    # порівнюємо кореляцію і згортку на несиметричному ядрі
    correlation = filters.imfilter(
        image, kernel, mode="corr", boundary="replicate", size="same"
    )
    convolution = filters.imfilter(
        image, kernel, mode="conv", boundary="replicate", size="same"
    )
    difference = np.abs(correlation - convolution)
    max_abs_diff = float(difference.max())
    mean_abs_diff = float(difference.mean())

    corr_path = save_result_image(
        "corr_asymmetric", response_visual(correlation, symmetric=True)
    )
    conv_path = save_result_image(
        "conv_asymmetric", response_visual(convolution, symmetric=True)
    )

    # відгук на одиничний імпульс показує як ядро лежить у відгуку
    impulse = np.zeros((9, 9), dtype=np.float64)
    impulse[4, 4] = 255.0
    impulse_corr = filters.imfilter(
        impulse, kernel, mode="corr", boundary="zero", size="full"
    )
    impulse_conv = filters.imfilter(
        impulse, kernel, mode="conv", boundary="zero", size="full"
    )

    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    kernel_image = kernel_axis(
        axes[0, 0],
        kernel,
        f"Несиметричне ядро 3×3\nΣ = {kernel.sum():.2f}",
    )
    figure.colorbar(kernel_image, ax=axes[0, 0], fraction=0.046, pad=0.03)

    limit = float(
        np.percentile(
            np.abs(np.concatenate([correlation.ravel(), convolution.ravel()])), 99.5
        )
    )
    limit = max(limit, 1e-9)
    for axis, data, title in (
        (axes[0, 1], correlation, "Кореляція ('corr')\nдвополярна шкала"),
        (axes[0, 2], convolution, "Згортка ('conv')\nдвополярна шкала"),
    ):
        shown = axis.imshow(
            data, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
        )
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)

    diff_image = axes[1, 0].imshow(
        difference,
        cmap="magma",
        vmin=0.0,
        vmax=float(np.percentile(difference, 99.5)) or 1.0,
        interpolation="nearest",
    )
    axes[1, 0].set_title(
        f"|кореляція - згортка|\nmax = {max_abs_diff:.1f}, "
        f"середнє = {mean_abs_diff:.2f}",
        fontsize=10,
    )
    axes[1, 0].axis("off")
    figure.colorbar(diff_image, ax=axes[1, 0], fraction=0.046, pad=0.03)

    for axis, response, title in (
        (
            axes[1, 1],
            impulse_corr,
            "Повний відгук на імпульс: 'corr'\nядро повернуте на 180°",
        ),
        (
            axes[1, 2],
            impulse_conv,
            "Повний відгук на імпульс: 'conv'\nядро без повороту",
        ),
    ):
        impulse_limit = float(np.max(np.abs(response))) or 1.0
        shown = axis.imshow(
            response,
            cmap="RdBu_r",
            vmin=-impulse_limit,
            vmax=impulse_limit,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)

    figure.suptitle(
        "Рисунок 3. Кореляція та згортка з несиметричним ядром",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig03_correlation_vs_convolution.png")

    return {
        "kernel": filters.kernel_metadata(kernel),
        "correlation_image": corr_path,
        "convolution_image": conv_path,
        "max_abs_difference": max_abs_diff,
        "mean_abs_difference": mean_abs_diff,
    }


def same_full_experiment(image, kernel, data_max):
    # порівнюємо розміри виходу same і full
    same = filters.imfilter(
        image, kernel, mode="corr", boundary="replicate", size="same"
    )
    full = filters.imfilter(
        image, kernel, mode="corr", boundary="replicate", size="full"
    )
    same_uint8 = to_uint8(same)
    full_uint8 = to_uint8(full)
    same_path = save_result_image("average_same", same_uint8)
    full_path = save_result_image("average_full", full_uint8)

    h, w = kernel.shape
    offset_y, offset_x = h // 2, w // 2
    image_h, image_w = image.shape

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    image_axis(
        axes[0],
        same_uint8,
        f"'same'\n{image_h}×{image_w}",
        vmin=0.0,
        vmax=data_max,
    )
    image_axis(
        axes[1],
        full_uint8,
        f"'full'\n{full.shape[0]}×{full.shape[1]}",
        vmin=0.0,
        vmax=data_max,
    )
    axes[1].add_patch(
        Rectangle(
            (offset_x - 0.5, offset_y - 0.5),
            image_w,
            image_h,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.2,
        )
    )
    axes[1].text(
        0.02,
        -0.04,
        "червона рамка: центральна частина, що відповідає 'same'",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#d62728",
    )

    positions = np.arange(4)
    values = [image_h, image_w, full.shape[0], full.shape[1]]
    labels = ["same\nрядки", "same\nстовпці", "full\nрядки", "full\nстовпці"]
    bars = axes[2].bar(
        positions,
        values,
        color=["#4c72b0", "#4c72b0", "#dd8452", "#dd8452"],
    )
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylim(min(values) - 40, max(values) + 35)
    axes[2].set_ylabel("Кількість елементів")
    axes[2].set_title("Розміри вихідного масиву", fontsize=10)
    axes[2].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].text(
        0.5,
        0.05,
        f"full = same + ({h - 1}, {w - 1})",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7"},
    )

    figure.suptitle(
        "Рисунок 4. Опції розміру imfilter: 'same' та 'full' (усереднення 5×5)",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig04_same_vs_full.png")

    return {
        "kernel": "average 5x5",
        "input_shape": [image_h, image_w],
        "same_shape": [int(same.shape[0]), int(same.shape[1])],
        "full_shape": [int(full.shape[0]), int(full.shape[1])],
        "same_image": same_path,
        "full_image": full_path,
    }


def build_fspecial_kernels():
    # будуємо всі стандартні ядра
    kernels = {}
    for spec in KERNEL_SPECS:
        kernels[spec["name"]] = filters.fspecial(spec["matlab"], *spec["args"])
    return kernels


def figure_fspecial_kernels(kernels):
    # рисунок 5: теплові карти всіх стандартних ядер
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, spec in zip(axes.ravel(), KERNEL_SPECS):
        kernel = kernels[spec["name"]]
        title = (
            f"{spec['label']} ({spec['matlab']})\n"
            f"{spec['call']}\n"
            f"Σ = {kernel.sum():.4f}"
        )
        image = kernel_axis(axis, kernel, title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        "Рисунок 5. Ядра, побудовані аналогом fspecial",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig05_fspecial_kernels.png")


def fspecial_experiments(image, kernels):
    # застосовуємо стандартні ядра до зображення
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    data_max = float(image.max())
    entries = {}

    for axis, spec in zip(axes.ravel(), KERNEL_SPECS):
        name = spec["name"]
        kernel = kernels[name]
        response = filters.imfilter(
            image, kernel, mode="corr", boundary="replicate", size="same"
        )
        entry = {
            "label": spec["label"],
            "call": spec["call"],
            "params": spec["params"],
            "kernel": filters.kernel_metadata(kernel),
        }

        if spec["highpass"]:
            # для високочастотних ядер показуємо знакозмінний відгук
            limit = max(float(np.percentile(np.abs(response), 99.5)), 1e-9)
            visual = response_visual(response, symmetric=True)
            path = save_result_image(f"fspecial_{name}", visual)
            entry["image"] = path
            entry["display"] = "знакозмінний відгук, двобічна шкала"
            image_axis(
                axis,
                response,
                f"{spec['label']}\nзнакозмінний відгук (двобічна шкала)",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
        else:
            result = to_uint8(response)
            path = save_result_image(f"fspecial_{name}", result)
            entry["image"] = path
            entry["display"] = "обрізано до uint8 0...255"
            entry["metrics_vs_input"] = metrics_vs(image, result)
            image_axis(
                axis,
                result,
                f"{spec['label']}\n{spec['call']}",
                vmin=0.0,
                vmax=data_max,
            )
        entries[name] = entry

    figure.suptitle(
        "Рисунок 6. Результати фільтрації стандартними ядрами fspecial\n"
        "(кореляція, гранична умова replicate)",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig06_fspecial_results.png")
    return entries


def custom_filter_experiments(image):
    # власне ядро підвищення різкості і гауссів фільтр 11 на 11
    sharpening = filters.sharpening_kernel(strength=0.5)
    custom_gaussian = filters.fspecial("gaussian", 11, 2.0)

    sharpened_response = filters.imfilter(
        image, sharpening, mode="corr", boundary="replicate", size="same"
    )
    gaussian_response = filters.imfilter(
        image, custom_gaussian, mode="corr", boundary="replicate", size="same"
    )
    sharpened = to_uint8(sharpened_response)
    gaussian_filtered = to_uint8(gaussian_response)
    sharpened_path = save_result_image("custom_sharpening", sharpened)
    gaussian_path = save_result_image("custom_gaussian", gaussian_filtered)

    # скільки деталей у зображенні: реакція на лапласіан
    detail_before = float(
        np.mean(
            np.abs(
                filters.imfilter(
                    image,
                    filters.laplacian_kernel(0.2),
                    mode="corr",
                    boundary="replicate",
                    size="same",
                )
            )
        )
    )
    detail_after = float(
        np.mean(
            np.abs(
                filters.imfilter(
                    sharpened,
                    filters.laplacian_kernel(0.2),
                    mode="corr",
                    boundary="replicate",
                    size="same",
                )
            )
        )
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    kernel_image = kernel_axis(
        axes[0, 0],
        sharpening,
        "Власне ядро підвищення різкості 5×5\nδ + 0.5·L5, Σ = 1",
    )
    figure.colorbar(kernel_image, ax=axes[0, 0], fraction=0.046, pad=0.03)
    sharpened_max = float(np.max(sharpened))
    image_axis(
        axes[0, 1],
        sharpened,
        f"Результат підвищення різкості\n(шкала 0-{sharpened_max:.0f})",
        vmin=0.0,
        vmax=sharpened_max,
    )
    kernel_image = kernel_axis(
        axes[1, 0],
        custom_gaussian,
        "Власний гауссів фільтр 11×11\nσ = 2.0, Σ = 1",
    )
    figure.colorbar(kernel_image, ax=axes[1, 0], fraction=0.046, pad=0.03)
    gaussian_max = float(np.max(gaussian_filtered))
    image_axis(
        axes[1, 1],
        gaussian_filtered,
        f"Результат гауссового фільтра\n11×11, σ = 2.0 (шкала 0-{gaussian_max:.0f})",
        vmin=0.0,
        vmax=gaussian_max,
    )
    figure.suptitle(
        "Рисунок 7. Власні фільтри: підвищення різкості та гауссів 11×11 із σ = 2.0",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig07_custom_filters.png")

    return {
        "sharpening": {
            "formula": "delta + 0.5 * laplacian5",
            "params": {"size": [5, 5], "strength": 0.5},
            "kernel": filters.kernel_metadata(sharpening),
            "image": sharpened_path,
            "detail_energy_before": detail_before,
            "detail_energy_after": detail_after,
            "detail_energy_ratio": detail_after / max(detail_before, 1e-12),
        },
        "gaussian_11x11_sigma2": {
            "formula": "exp(-(x² + y²) / (2·σ²)), нормовано до суми 1",
            "params": {"size": [11, 11], "sigma": 2.0},
            "kernel": filters.kernel_metadata(custom_gaussian),
            "image": gaussian_path,
            "metrics_vs_input": metrics_vs(image, gaussian_filtered),
        },
    }


def add_gaussian_noise(image, sigma, rng):
    # додаємо гауссів шум з нульовим середнім
    noisy = image.astype(np.float64) + rng.normal(0.0, sigma, image.shape)
    return to_uint8(noisy)


def add_salt_pepper_noise(image, amount, rng):
    # частину пікселів міняємо на 0 або 255
    if not 0.0 <= amount <= 1.0:
        raise ValueError("Частка шуму має бути в діапазоні 0...1.")
    noisy = image.copy()
    total = int(round(amount * image.size))
    if total == 0:
        return noisy
    indices = rng.choice(image.size, size=total, replace=False)
    half = total // 2
    flat = noisy.reshape(-1)
    flat[indices[:half]] = 255
    flat[indices[half:]] = 0
    return noisy


def bilateral_filter(image):
    # білатеральний фільтр, нелінійний
    normalized = image.astype(np.float64) / 255.0
    filtered = denoise_bilateral(
        normalized,
        win_size=5,
        sigma_color=0.12,
        sigma_spatial=2.0,
        channel_axis=None,
        mode="edge",
    )
    return to_uint8(filtered * 255.0)


def noise_figure(clean, noisy, results, metrics, title, filename):
    # спільний рисунок порівняння для одного типу шуму
    panels = [
        (clean, "Оригінал (еталон)"),
        (noisy, f"Зашумлене\nPSNR = {metrics['noisy']['psnr']:.2f} дБ"),
    ]
    labels = {
        "average": "Усереднення 5×5 (лінійний)",
        "gaussian": "Гауссів 9×9, σ = 0.8 (лінійний)",
        "median": "Медіанний 5×5 (нелінійний)",
        "bilateral": "Білатеральний (нелінійний)",
    }
    for name in ("average", "gaussian", "median", "bilateral"):
        entry = metrics["filters"][name]
        panels.append(
            (
                results[name],
                f"{labels[name]}\n"
                f"PSNR = {entry['psnr']:.2f} дБ, SSIM = {entry['ssim']:.3f}",
            )
        )

    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    for axis, (panel, panel_title) in zip(axes.ravel(), panels):
        image_axis(
            axis,
            panel,
            panel_title,
            vmin=0.0,
            vmax=DISPLAY_MAX,
        )
    figure.suptitle(f"{title}\n(спільна шкала відображення 0-160)", fontsize=12)
    save_figure(figure, OUT_DIR / filename)


def noise_experiments(image):
    # фільтруємо зашумлені зображення різними фільтрами
    rng = np.random.default_rng(SEED)
    average_kernel = filters.fspecial("average", 5)
    gaussian_kernel = filters.fspecial("gaussian", 9, 0.8)

    denoisers = [
        (
            "average",
            "linear",
            "усереднення 5×5",
            {"kernel": "average 5x5", "boundary": "replicate"},
            lambda data: filters.imfilter(
                data, average_kernel, mode="corr", boundary="replicate", size="same"
            ),
        ),
        (
            "gaussian",
            "linear",
            "гауссів 9×9, σ = 0.8",
            {"kernel": "gaussian 9x9", "sigma": 0.8, "boundary": "replicate"},
            lambda data: filters.imfilter(
                data, gaussian_kernel, mode="corr", boundary="replicate", size="same"
            ),
        ),
        (
            "median",
            "nonlinear",
            "медіанний 5×5",
            {"size": 5, "boundary": "nearest"},
            lambda data: ndimage.median_filter(data, size=5, mode="nearest"),
        ),
        (
            "bilateral",
            "nonlinear",
            "білатеральний",
            {
                "win_size": 5,
                "sigma_color": 0.12,
                "sigma_spatial": 2.0,
                "mode": "edge",
            },
            bilateral_filter,
        ),
    ]

    noise_types = {
        "gaussian": {
            "label": "Гауссів шум",
            "parameters": {"sigma": 18.0},
            "noisy": add_gaussian_noise(image, 18.0, rng),
        },
        "salt_pepper": {
            "label": "Шум «сіль і перець»",
            "parameters": {"amount": 0.05},
            "noisy": add_salt_pepper_noise(image, 0.05, rng),
        },
    }

    result = {
        "seed": SEED,
        "clean": "data/input.png",
        "types": {},
    }

    for key, item in noise_types.items():
        noisy = item["noisy"]
        noisy_path = save_result_image(f"noisy_{key}", noisy)
        noisy_metrics = metrics_vs(image, noisy)
        entry = {
            "label": item["label"],
            "parameters": item["parameters"],
            "noisy": {"image": noisy_path, **noisy_metrics},
            "filters": {},
        }

        filter_images = {}
        for name, kind, filter_label, params, function in denoisers:
            filtered = to_uint8(function(noisy))
            filter_images[name] = filtered
            path = save_result_image(f"denoise_{key}_{name}", filtered)
            metrics = metrics_vs(image, filtered)
            metrics["psnr_gain_vs_noisy"] = metrics["psnr"] - noisy_metrics["psnr"]
            entry["filters"][name] = {
                "type": kind,
                "label": filter_label,
                "parameters": params,
                "image": path,
                **metrics,
            }

        best = max(entry["filters"], key=lambda name: entry["filters"][name]["psnr"])
        entry["best_psnr_filter"] = best
        entry["best_psnr_value"] = entry["filters"][best]["psnr"]
        result["types"][key] = entry

        if key == "gaussian":
            title = (
                "Рисунок 8. Лінійна та нелінійна фільтрація гауссового шуму "
                "(σ = 18)"
            )
            filename = "fig08_gaussian_noise_comparison.png"
        else:
            title = (
                "Рисунок 9. Лінійна та нелінійна фільтрація шуму «сіль і перець» "
                "(5 % пікселів)"
            )
            filename = "fig09_salt_pepper_noise_comparison.png"

        noise_figure(
            image,
            noisy,
            filter_images,
            {"noisy": noisy_metrics, "filters": entry["filters"]},
            title,
            filename,
        )

    figure_metrics_comparison(result)
    return result


def figure_metrics_comparison(noise_result):
    # рисунок 10: стовпчикові діаграми psnr і ssim
    names = ["noisy", "average", "gaussian", "median", "bilateral"]
    labels = ["зашумлене", "усереднення", "гауссів", "медіанний", "білатеральний"]
    colors = {
        "gaussian": "#4c72b0",
        "salt_pepper": "#dd8452",
    }
    display_labels = {
        "gaussian": "Гауссів шум",
        "salt_pepper": "Сіль і перець",
    }

    figure, axes = plt.subplots(1, 3, figsize=(21, 6))
    width = 0.36
    x = np.arange(len(names))

    for index, (key, item) in enumerate(noise_result["types"].items()):
        psnr_values = [item["noisy"]["psnr"]] + [
            item["filters"][name]["psnr"] for name in names[1:]
        ]
        ssim_values = [item["noisy"]["ssim"]] + [
            item["filters"][name]["ssim"] for name in names[1:]
        ]
        edge_values = [item["noisy"]["edge_correlation"]] + [
            item["filters"][name]["edge_correlation"] for name in names[1:]
        ]
        offset = (index - 0.5) * width
        bars_psnr = axes[0].bar(
            x + offset,
            psnr_values,
            width,
            label=display_labels[key],
            color=colors[key],
        )
        axes[1].bar(
            x + offset,
            ssim_values,
            width,
            label=display_labels[key],
            color=colors[key],
        )
        bars_edge = axes[2].bar(
            x + offset,
            edge_values,
            width,
            label=display_labels[key],
            color=colors[key],
        )
        for bar, value in zip(bars_psnr, psnr_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.35,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        for bar, value in zip(bars_edge, edge_values):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    axes[0].set_title("Пікове відношення сигнал/шум (PSNR)", fontsize=10)
    axes[0].set_ylabel("PSNR, дБ")
    axes[0].set_ylim(10.0, 45.0)
    axes[1].set_title("Структурна подібність (SSIM)", fontsize=10)
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0.0, 1.05)
    axes[2].set_title("Кореляція градієнтів (збереження границь)", fontsize=10)
    axes[2].set_ylabel("Кореляція модулів градієнтів")
    axes[2].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Рисунок 10. Порівняння лінійних і нелінійних фільтрів за типами шуму",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig10_metrics_comparison.png")


def main():
    # робимо всі етапи роботи і пишемо результати
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image = load_gray(DATA_DIR / "input.png")
    checkerboard = load_gray(DATA_DIR / "checkerboard.png")
    vmin, vmax = display_window(image)
    data_max = float(image.max())

    print("1. вхідне зображення")
    figure_input(image, vmin, vmax)

    average5 = filters.fspecial("average", 5)
    asymmetric = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    print("2. граничні умови imfilter")
    boundary_result = boundary_experiment(checkerboard, average5)

    print("3. кореляція і згортка")
    corr_conv_result = correlation_convolution_experiment(image, asymmetric)

    print("4. розміри same і full")
    same_full_result = same_full_experiment(image, average5, data_max)

    print("5. стандартні ядра fspecial")
    kernels = build_fspecial_kernels()
    figure_fspecial_kernels(kernels)
    fspecial_result = fspecial_experiments(image, kernels)

    print("6. власні фільтри")
    custom_result = custom_filter_experiments(image)

    print("7. шум і різні фільтри")
    noise_result = noise_experiments(image)

    results = {
        "work": "praktychna-04",
        "title": "Практична робота 4. Обробка зображень цифровими фільтрами",
        "input": {
            "path": "data/input.png",
            "stats": image_stats(image),
            "display_window": [vmin, vmax],
        },
        "checkerboard": {
            "path": "data/checkerboard.png",
            "stats": image_stats(checkerboard),
        },
        "imfilter": {
            "average_kernel": filters.kernel_metadata(average5),
            "asymmetric_kernel": filters.kernel_metadata(asymmetric),
            "boundary_comparison": boundary_result,
            "correlation_vs_convolution": corr_conv_result,
            "same_vs_full": same_full_result,
        },
        "fspecial": fspecial_result,
        "custom_filters": custom_result,
        "noise_experiments": noise_result,
        "outputs": {
            "figures": sorted(rel_path(path) for path in OUT_DIR.glob("*.png")),
            "images": sorted(rel_path(path) for path in IMAGE_DIR.glob("*.png")),
        },
    }

    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    print("готово, результати в out/results.json")


if __name__ == "__main__":
    main()
