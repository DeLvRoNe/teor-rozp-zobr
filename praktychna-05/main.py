# практична робота 5. перетворення фур'є для обробки зображень
# запуск: python main.py

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_bilateral

import fourier
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

# utils перемикає matplotlib на Agg, тому pyplot беремо після нього
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
IMAGE_DIR = OUT_DIR / "images"

# прибираємо попередження matplotlib про tight_layout
warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout.*",
    category=UserWarning,
)

SEED = 20241105
NOISE_SIGMA = 20.0
DESIGN_SIZE = 31
CUTOFF = 0.20
CUSTOM_INNER = 0.14
CUSTOM_OUTER = 0.30
NOISE_INNER = 0.10
NOISE_OUTER = 0.24
DISPLAY_MAX = 255.0


# --- допоміжні функції ---


def rel_path(path):
    # шлях відносно папки практичної роботи
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def save_result_image(name, array):
    # зберігаємо окреме зображення в out/images
    path = IMAGE_DIR / f"{name}.png"
    save_gray(path, array)
    return rel_path(path)


def image_stats(array):
    # основні статистики зображення
    data = np.asarray(array, dtype=np.float64)
    return {
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def metrics_vs(reference, test):
    # mse, psnr, ssim і збереження границь
    data = {
        "mse": float(mse(reference, test)),
        "psnr": float(psnr(reference, test)),
        "ssim": float(ssim(reference, test)),
    }
    if not np.isfinite(data["psnr"]):
        data["psnr"] = 99.0
    data.update(fourier.edge_metrics(reference, test))
    return data


def evaluate_filter(reference, output):
    # метрики плюс енергія високих частот і лапласіана
    result = metrics_vs(reference, output)
    result["highpass_energy"] = fourier.highpass_energy(output)
    result["laplacian_energy"] = fourier.laplacian_energy(output)
    return result


def add_energy_ratios(metrics, reference_metrics):
    # відносні енергетичні показники відносно входу
    metrics["highpass_energy_ratio"] = float(
        metrics["highpass_energy"]
        / max(reference_metrics["highpass_energy"], 1e-12)
    )
    metrics["laplacian_energy_ratio"] = float(
        metrics["laplacian_energy"]
        / max(reference_metrics["laplacian_energy"], 1e-12)
    )
    return metrics


def response_db(response, floor_db=-60.0):
    # переводимо ачх у децибели
    magnitude = np.abs(np.asarray(response))
    floor = 10.0 ** (floor_db / 20.0)
    return np.clip(20.0 * np.log10(np.maximum(magnitude, floor)), floor_db, 3.0)


def response_stats(response, radius, cutoff):
    # показники смуги пропускання та затримування
    data = np.asarray(response, dtype=np.float64)
    center = (data.shape[0] // 2, data.shape[1] // 2)
    passband = data[radius <= cutoff]
    stopband = data[radius >= max(0.45, cutoff + 0.20)]
    stopband_abs_max = float(np.max(np.abs(stopband))) if stopband.size else 0.0
    if stopband_abs_max > 0.0:
        attenuation = 20.0 * np.log10(max(stopband_abs_max, 1e-15))
    else:
        attenuation = -300.0
    return {
        "dc_gain": float(data[center]),
        "max": float(data.max()),
        "min": float(data.min()),
        "overshoot": float(max(0.0, data.max() - 1.0)),
        "undershoot": float(min(0.0, data.min())),
        "passband_mean": float(passband.mean()),
        "stopband_max_abs": stopband_abs_max,
        "stopband_attenuation_db": float(attenuation),
    }


def add_gaussian_noise(image, sigma, rng):
    # адитивний гауссів шум
    noisy = image.astype(np.float64) + rng.normal(0.0, sigma, image.shape)
    return fourier.to_uint8(noisy)


def bilateral_filter(image):
    # білатеральний фільтр, зберігає границі
    normalized = image.astype(np.float64) / 255.0
    filtered = denoise_bilateral(
        normalized,
        win_size=5,
        sigma_color=0.12,
        sigma_spatial=2.0,
        channel_axis=None,
        mode="edge",
    )
    return fourier.to_uint8(filtered * 255.0)


def json_ready(value):
    # numpy-типи переводимо у звичайні типи python
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return rel_path(value)
    return value


# --- рисунки ---


def figure_input(image):
    # рисунок 1: вхідне зображення та гістограма
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    image_axis(
        axes[0],
        image,
        f"а) вхідне зображення\n{image.shape[1]}×{image.shape[0]}, uint8",
        vmin=0.0,
        vmax=DISPLAY_MAX,
    )
    histogram_axis(
        axes[1],
        image,
        "б) гістограма яскравості",
        bins=256,
        value_range=(0, 256),
    )
    figure.suptitle(
        "Рисунок 1. Вхідне напівтонове зображення", fontsize=12
    )
    save_figure(figure, OUT_DIR / "fig01_input_image.png")


def figure_spectrum(image, log_magnitude, phase):
    # рисунок 2: спектр, фаза, поверхня і профіль
    rows, columns = image.shape
    figure = plt.figure(figsize=(17, 12))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    axis_log = figure.add_subplot(grid[0, 0])
    shown = axis_log.imshow(
        log_magnitude,
        cmap="magma",
        interpolation="nearest",
        origin="upper",
    )
    axis_log.set_title(
        "а) центрований лог-спектр log(1 + |F|)", fontsize=10
    )
    axis_log.axis("off")
    axis_log.annotate(
        "DC (0, 0)",
        xy=(columns / 2.0, rows / 2.0),
        xytext=(columns / 2.0 + 55.0, rows / 2.0 + 55.0),
        fontsize=9,
        color="white",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
    )
    figure.colorbar(shown, ax=axis_log, fraction=0.046, pad=0.03)

    axis_phase = figure.add_subplot(grid[0, 1])
    shown_phase = axis_phase.imshow(
        phase,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        interpolation="nearest",
    )
    axis_phase.set_title("б) фаза спектра ∠F, рад", fontsize=10)
    axis_phase.axis("off")
    figure.colorbar(shown_phase, ax=axis_phase, fraction=0.046, pad=0.03)

    axis_mesh = figure.add_subplot(grid[1, 0], projection="3d")
    step = 4
    fy, fx = fourier.freqspace(image.shape)
    fy_sampled = fy[::step]
    fx_sampled = fx[::step]
    magnitude_sampled = log_magnitude[::step, ::step]
    grid_x, grid_y = np.meshgrid(fx_sampled, fy_sampled)
    axis_mesh.plot_surface(
        grid_x,
        grid_y,
        magnitude_sampled,
        cmap="viridis",
        rstride=4,
        cstride=4,
        linewidth=0,
        antialiased=True,
    )
    axis_mesh.set_xlabel("$f_x$", fontsize=9)
    axis_mesh.set_ylabel("$f_y$", fontsize=9)
    axis_mesh.set_zlabel("log(1 + |F|)", fontsize=9)
    axis_mesh.set_title("в) 3D-поверхня лог-спектра", fontsize=10)
    axis_mesh.view_init(elev=38.0, azim=-58.0)

    axis_profile = figure.add_subplot(grid[1, 1])
    centers, values = fourier.radial_profile(log_magnitude, bins=90)
    axis_profile.plot(centers, values, color="#2a6f97", linewidth=1.8)
    axis_profile.set_xlabel("Радіус нормованої частоти $r$")
    axis_profile.set_ylabel("Середній log(1 + |F|)")
    axis_profile.set_title("г) радіальний профіль лог-спектра", fontsize=10)
    axis_profile.grid(alpha=0.25)
    axis_profile.annotate(
        f"максимум при r = 0\nlog(1 + DC) = {values[0]:.2f}",
        xy=(centers[0], values[0]),
        xytext=(0.18, 0.82),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )

    figure.suptitle(
        "Рисунок 2. Спектр вхідного зображення (двовимірне дпф)",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig02_spectrum.png")


def figure_desired_response(desired, cutoff):
    # рисунок 3: бажана ачх нч-фільтра
    size = desired.shape[0]
    fy, fx = fourier.freqspace((size, size))
    figure = plt.figure(figsize=(18, 5.8))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15])

    axis_map = figure.add_subplot(grid[0, 0])
    shown = axis_map.imshow(
        desired,
        extent=(-1.0, 1.0, -1.0, 1.0),
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis_map.add_patch(
        Circle(
            (0.0, 0.0),
            cutoff,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.8,
        )
    )
    axis_map.set_xlabel("$f_x$ (нормовано до найквіста)")
    axis_map.set_ylabel("$f_y$ (нормовано до найквіста)")
    axis_map.set_title(
        f"а) бажаний відгук на сітці {size}×{size}", fontsize=10
    )
    figure.colorbar(shown, ax=axis_map, fraction=0.046, pad=0.03)

    axis_row = figure.add_subplot(grid[0, 1])
    axis_row.step(
        fx,
        desired[size // 2, :],
        where="mid",
        color="#1f77b4",
        linewidth=1.8,
        label="відліки Hd",
    )
    axis_row.axvline(cutoff, color="#d62728", linestyle="--", linewidth=1.2)
    axis_row.axvline(-cutoff, color="#d62728", linestyle="--", linewidth=1.2)
    axis_row.set_xlim(-1.0, 1.0)
    axis_row.set_ylim(-0.05, 1.05)
    axis_row.set_xlabel("$f_x$")
    axis_row.set_ylabel("$H_d$")
    axis_row.set_title("б) центральний переріз $f_y = 0$", fontsize=10)
    axis_row.grid(alpha=0.25)
    axis_row.legend(fontsize=8, loc="lower center")

    axis_surface = figure.add_subplot(grid[0, 2], projection="3d")
    grid_x, grid_y = np.meshgrid(fx, fy)
    axis_surface.plot_surface(
        grid_x,
        grid_y,
        desired,
        cmap="viridis",
        rstride=1,
        cstride=1,
        linewidth=0.2,
        antialiased=True,
    )
    axis_surface.set_xlabel("$f_x$", fontsize=9)
    axis_surface.set_ylabel("$f_y$", fontsize=9)
    axis_surface.set_zlabel("$H_d$", fontsize=9)
    axis_surface.set_title("в) 3D-подання бажаного відгуку", fontsize=10)
    axis_surface.view_init(elev=32.0, azim=-60.0)

    figure.suptitle(
        "Рисунок 3. Бажана ачх кругового нч-фільтра "
        f"(зріз {cutoff:.2f})",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig03_desired_response.png")


def figure_kernels(kernels, windows, hamming_1d):
    # рисунок 4: ядра фільтрів та вікна
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    descriptions = {
        "fsamp2": "fsamp2\nбез вікна (прямокутне)",
        "fwind1": "fwind1\nсепарабельне хеммінга",
        "fwind2": "fwind2\nрадіальне ханна",
    }
    for axis, name in zip(axes[0], ("fsamp2", "fwind1", "fwind2")):
        kernel = kernels[name]
        limit = max(float(np.max(np.abs(kernel))), 1e-9)
        shown = axis.imshow(
            kernel,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(
            f"{descriptions[name]}\n"
            f"Σ = {kernel.sum():.4f}, min = {kernel.min():.4f}, "
            f"max = {kernel.max():.4f}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)

    axis_windows = axes[1, 0]
    positions = np.linspace(-1.0, 1.0, hamming_1d.size)
    axis_windows.plot(
        positions, hamming_1d, color="#1f77b4", linewidth=1.8, label="хеммінга 1D"
    )
    axis_windows.plot(
        positions,
        windows["fwind2"][windows["fwind2"].shape[0] // 2, :],
        color="#d62728",
        linewidth=1.8,
        label="радіальне ханна (переріз)",
    )
    axis_windows.set_xlabel("Нормована координата вікна")
    axis_windows.set_ylabel("Значення вікна")
    axis_windows.set_title("г) одновимірні вікна", fontsize=10)
    axis_windows.grid(alpha=0.25)
    axis_windows.legend(fontsize=8)

    for axis, name, label in (
        (axes[1, 1], "fwind1", "д) вікно fwind1 = hamming ⊗ hamming"),
        (axes[1, 2], "fwind2", "е) вікно fwind2 (радіальне ханна)"),
    ):
        window = windows[name]
        shown = axis.imshow(
            window, cmap="viridis", interpolation="nearest"
        )
        axis.set_title(
            f"{label}\nmin = {window.min():.4f}, max = {window.max():.4f}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03)

    figure.suptitle(
        "Рисунок 4. Ядра fsamp2/fwind1/fwind2 та вікна для синтезу",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig04_kernels.png")


def figure_filter_responses(responses, desired_full, radius_full, cutoff):
    # рисунок 5: ачх фільтрів у 2d і радіальні перерізи
    figure = plt.figure(figsize=(20, 12))
    grid = figure.add_gridspec(2, 4, height_ratios=[1.0, 1.05])

    top_axes = []
    for index, (name, item) in enumerate(responses.items()):
        axis = figure.add_subplot(grid[0, index])
        top_axes.append(axis)
        shown = axis.imshow(
            response_db(item["response"]),
            extent=(-1.0, 1.0, -1.0, 1.0),
            origin="lower",
            cmap="viridis",
            vmin=-60.0,
            vmax=3.0,
            interpolation="nearest",
        )
        stats = response_stats(item["response"], radius_full, cutoff)
        axis.set_title(
            f"{item['label']}\n"
            f"min = {stats['min']:.3f}, max = {stats['max']:.3f}, "
            f"затрим. = {stats['stopband_attenuation_db']:.1f} дБ",
            fontsize=9,
        )
        axis.set_xlabel("$f_x$", fontsize=8)
        axis.set_ylabel("$f_y$", fontsize=8)
    figure.colorbar(
        shown,
        ax=top_axes,
        fraction=0.025,
        pad=0.01,
        label="20·log10|H|, дБ",
    )

    axis_linear = figure.add_subplot(grid[1, :2])
    for name, item in responses.items():
        centers, values = fourier.radial_profile(item["response"], bins=90)
        axis_linear.plot(centers, values, linewidth=1.7, label=item["label"])
    centers_desired, values_desired = fourier.radial_profile(
        desired_full, bins=90
    )
    axis_linear.plot(
        centers_desired,
        values_desired,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="бажаний відгук",
    )
    axis_linear.set_xlim(0.0, 1.0)
    axis_linear.set_ylim(-0.08, 1.35)
    axis_linear.set_xlabel("Радіус нормованої частоти $r$")
    axis_linear.set_ylabel("Амплітуда H")
    axis_linear.set_title("в) радіальні перерізи ачх (лінійна шкала)", fontsize=10)
    axis_linear.grid(alpha=0.25)
    axis_linear.legend(fontsize=8, ncol=2)

    axis_db = figure.add_subplot(grid[1, 2:])
    for name, item in responses.items():
        centers, values = fourier.radial_profile(
            response_db(item["response"]), bins=90
        )
        axis_db.plot(centers, values, linewidth=1.7, label=item["label"])
    centers_desired, values_desired = fourier.radial_profile(
        response_db(desired_full), bins=90
    )
    axis_db.plot(
        centers_desired,
        values_desired,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="бажаний відгук",
    )
    axis_db.set_xlim(0.0, 1.0)
    axis_db.set_ylim(-65.0, 5.0)
    axis_db.set_xlabel("Радіус нормованої частоти $r$")
    axis_db.set_ylabel("20·log10|H|, дБ")
    axis_db.set_title("г) радіальні перерізи ачх (децибели)", fontsize=10)
    axis_db.grid(alpha=0.25)
    axis_db.legend(fontsize=8, ncol=2)

    figure.suptitle(
        "Рисунок 5. Частотні характеристики синтезованих фільтрів",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig05_filter_responses.png")


def figure_filtered_images(image, filtered):
    # рисунок 6: результати фільтрації
    figure, axes = plt.subplots(2, 3, figsize=(17, 11))
    image_axis(
        axes[0, 0],
        image,
        "Оригінал",
        vmin=0.0,
        vmax=DISPLAY_MAX,
    )
    positions = [
        (axes[0, 1], "fsamp2"),
        (axes[0, 2], "fwind1"),
        (axes[1, 0], "fwind2"),
        (axes[1, 1], "custom"),
    ]
    for axis, name in positions:
        item = filtered[name]
        metrics = item["metrics"]
        image_axis(
            axis,
            item["output"],
            f"{item['label']}\n"
            f"PSNR = {metrics['psnr']:.2f} дБ, SSIM = {metrics['ssim']:.3f}",
            vmin=0.0,
            vmax=DISPLAY_MAX,
        )

    difference = (
        filtered["custom"]["output"].astype(np.float64)
        - image.astype(np.float64)
    )
    limit = float(np.percentile(np.abs(difference), 99.5))
    limit = max(limit, 1e-9)
    shown = axes[1, 2].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axes[1, 2].set_title(
        "власний фільтр - оригінал\n"
        f"двобічна шкала ±{limit:.1f} рівня яскравості",
        fontsize=10,
    )
    axes[1, 2].axis("off")
    figure.colorbar(shown, ax=axes[1, 2], fraction=0.046, pad=0.03)

    figure.suptitle(
        "Рисунок 6. Результати фільтрації нч-фільтрами "
        "(спільна шкала 0-255)",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig06_filtered_images.png")


def figure_noise_comparison(clean, noise_result):
    # рисунок 7: зашумлене зображення та результати фільтрації
    noisy_image = load_gray(BASE_DIR / noise_result["noisy"]["image"])
    entries = [
        ("Оригінал (еталон)", clean, None),
        (
            f"Гауссів шум σ = {noise_result['sigma']:g}",
            noisy_image,
            noise_result["noisy"]["metrics_vs_clean"],
        ),
    ]
    for key, label in (
        ("linear_frequency", "Лінійний частотний НЧ"),
        ("median", "Медіанний 5×5"),
        ("bilateral", "Білатеральний"),
    ):
        entry = noise_result["filters"][key]
        entries.append(
            (
                label,
                load_gray(BASE_DIR / entry["image"]),
                entry["metrics_vs_clean"],
            )
        )

    rows, columns = clean.shape
    half = 96
    center_y, center_x = rows // 2, columns // 2
    crop_slice = (
        slice(center_y - half, center_y + half),
        slice(center_x - half, center_x + half),
    )

    figure, axes = plt.subplots(2, 5, figsize=(23, 10))
    for index, (label, panel, panel_metrics) in enumerate(entries):
        if panel_metrics is None:
            title = label
        else:
            title = (
                f"{label}\n"
                f"PSNR = {panel_metrics['psnr']:.2f} дБ, "
                f"SSIM = {panel_metrics['ssim']:.3f}"
            )
        image_axis(
            axes[0, index],
            panel,
            title,
            vmin=0.0,
            vmax=DISPLAY_MAX,
        )
        image_axis(
            axes[1, index],
            panel[crop_slice],
            f"фрагмент 192×192\n{label}",
            vmin=0.0,
            vmax=DISPLAY_MAX,
        )

    figure.suptitle(
        "Рисунок 7. Лінійна та нелінійна фільтрація гауссового шуму "
        f"(σ = {noise_result['sigma']:g})",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig07_noisy_comparison.png")


def figure_metrics_summary(noise_result):
    # рисунок 8: діаграми psnr, ssim, границь і енергії
    keys = ["noisy", "linear_frequency", "median", "bilateral"]
    labels = [
        "зашумлене",
        "лінійний\nчастотний",
        "медіанний\n5×5",
        "білатеральний",
    ]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
    entries = [noise_result["noisy"]] + [
        noise_result["filters"][key] for key in keys[1:]
    ]

    psnr_values = [entry["metrics_vs_clean"]["psnr"] for entry in entries]
    ssim_values = [entry["metrics_vs_clean"]["ssim"] for entry in entries]
    global_edges = [
        entry["metrics_vs_clean"]["gradient_correlation"] for entry in entries
    ]
    strong_edges = [
        entry["metrics_vs_clean"]["strong_edge_correlation"]
        for entry in entries
    ]
    highpass_values = [
        entry["metrics_vs_clean"]["highpass_energy"] for entry in entries
    ]

    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    x = np.arange(len(keys))

    panels = [
        (axes[0, 0], psnr_values, "а) PSNR, дБ", "PSNR, дБ"),
        (axes[0, 1], ssim_values, "б) SSIM", "SSIM"),
    ]
    for axis, values, title, ylabel in panels:
        bars = axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(values) * 0.015,
                f"{value:.3f}" if ylabel == "SSIM" else f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0, 0].set_ylim(0.0, max(psnr_values) * 1.18)
    axes[0, 1].set_ylim(0.0, 1.05)

    width = 0.38
    bars_global = axes[1, 0].bar(
        x - width / 2.0,
        global_edges,
        width,
        color="#4c72b0",
        label="глобальна кореляція",
    )
    bars_strong = axes[1, 0].bar(
        x + width / 2.0,
        strong_edges,
        width,
        color="#dd8452",
        label="сильні грані (10 %)",
    )
    axes[1, 0].set_xticks(x, labels, fontsize=8)
    axes[1, 0].set_ylabel("Кореляція модулів градієнтів")
    axes[1, 0].set_title("в) збереження границь", fontsize=10)
    axes[1, 0].set_ylim(-0.05, 1.0)
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    for bars in (bars_global, bars_strong):
        for bar in bars:
            value = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    bars_energy = axes[1, 1].bar(
        x, np.maximum(highpass_values, 1e-3), color=colors
    )
    axes[1, 1].set_xticks(x, labels, fontsize=8)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Середня енергія високих частот")
    axes[1, 1].set_title("г) високочастотна енергія (менше - краще)", fontsize=10)
    axes[1, 1].grid(axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars_energy, highpass_values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            max(value, 1e-3) * 1.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Рисунок 8. Порівняння лінійної та нелінійної фільтрації "
        "за метриками",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig08_metrics_summary.png")


def figure_radial_spectra(clean, filtered, noise_result):
    # рисунок 9: радіальні спектри потужності
    figure, axes = plt.subplots(1, 2, figsize=(17, 6.5))

    items = [("оригінал", clean)]
    for name in ("fsamp2", "fwind1", "fwind2", "custom"):
        items.append((filtered[name]["label"], filtered[name]["output"]))
    for label, panel in items:
        power = fourier.power_spectrum(panel)
        centers, values = fourier.radial_profile(power, bins=70)
        axes[0].semilogy(
            centers, np.maximum(values, 1e-3), linewidth=1.6, label=label
        )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Радіус нормованої частоти $r$")
    axes[0].set_ylabel("Середня потужність (лог. шкала)")
    axes[0].set_title("а) оригінал і стандартні нч-фільтри", fontsize=10)
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)

    noise_items = [
        ("оригінал", clean),
        (
            "зашумлене",
            load_gray(BASE_DIR / noise_result["noisy"]["image"]),
        ),
        (
            "лінійний частотний",
            load_gray(
                BASE_DIR
                / noise_result["filters"]["linear_frequency"]["image"]
            ),
        ),
        (
            "медіанний 5×5",
            load_gray(BASE_DIR / noise_result["filters"]["median"]["image"]),
        ),
        (
            "білатеральний",
            load_gray(
                BASE_DIR / noise_result["filters"]["bilateral"]["image"]
            ),
        ),
    ]
    for label, panel in noise_items:
        power = fourier.power_spectrum(panel)
        centers, values = fourier.radial_profile(power, bins=70)
        axes[1].semilogy(
            centers, np.maximum(values, 1e-3), linewidth=1.6, label=label
        )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Радіус нормованої частоти $r$")
    axes[1].set_ylabel("Середня потужність (лог. шкала)")
    axes[1].set_title("б) шум і результати відновлення", fontsize=10)
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "Рисунок 9. Радіальні спектри потужності",
        fontsize=12,
    )
    save_figure(figure, OUT_DIR / "fig09_radial_spectra.png")


def build_output_manifest():
    # перелік згенерованих файлів
    figures = sorted(rel_path(path) for path in OUT_DIR.glob("*.png"))
    images = sorted(rel_path(path) for path in IMAGE_DIR.glob("*.png"))
    return {
        "figures": figures,
        "images": images,
        "files": [rel_path(OUT_DIR / "results.json")],
    }


# --- основний сценарій ---


def main():
    # щоб український текст не ламався у консолі
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # читаємо картинку
    image = load_gray(DATA_DIR / "input.png")
    rows, columns = image.shape
    figure_input(image)
    save_result_image("input_gray", image)

    # будуємо спектр: прямий дпф, модуль і центрування
    image_float = image.astype(np.float64)
    spectrum = fourier.fft2(image_float)
    centered = fourier.fftshift(spectrum)
    log_magnitude = np.log1p(np.abs(centered))
    phase = np.angle(centered)
    reconstructed = np.real(np.fft.ifft2(spectrum))
    roundtrip_error = float(np.max(np.abs(reconstructed - image_float)))
    dc_value = float(np.abs(spectrum[0, 0]))
    pixel_sum = float(image_float.sum())
    figure_spectrum(image, log_magnitude, phase)
    save_result_image(
        "spectrum_log_magnitude",
        fourier.to_uint8(
            255.0 * log_magnitude / max(float(log_magnitude.max()), 1e-12)
        ),
    )

    # приклади частотної сітки
    even_example = fourier.freqspace((4, 4))[0]
    odd_example = fourier.freqspace((5, 5))[0]
    whole_example = fourier.freqspace_1d(4, whole=True)

    # синтезуємо ядра фільтрів
    desired, _ = fourier.desired_lowpass(
        (DESIGN_SIZE, DESIGN_SIZE), CUTOFF
    )
    kernel_fs = fourier.fsamp2(desired)
    hamming_1d = fourier.hamming_window(DESIGN_SIZE)
    window_fwind1 = np.outer(hamming_1d, hamming_1d)
    kernel_f1 = fourier.fwind1(desired, hamming_1d)
    window_fwind2 = fourier.radial_hann_window(DESIGN_SIZE)
    kernel_f2 = fourier.fwind2(desired, window_fwind2)
    kernels = {
        "fsamp2": kernel_fs,
        "fwind1": kernel_f1,
        "fwind2": kernel_f2,
    }
    figure_desired_response(desired, CUTOFF)
    figure_kernels(
        kernels,
        {"fwind1": window_fwind1, "fwind2": window_fwind2},
        hamming_1d,
    )

    # рахуємо ачх фільтрів
    pad_shape = fourier.paddedsize(image.shape)
    desired_full, radius_full = fourier.desired_lowpass(pad_shape, CUTOFF)
    responses = {}
    for name, kernel in kernels.items():
        responses[name] = {
            "label": name,
            "response": fourier.kernel_frequency_response(
                kernel, pad_shape
            ).real,
        }

    # власний нч-фільтр з косинусним переходом
    custom_response, _ = fourier.custom_lowpass_response(
        pad_shape, CUSTOM_INNER, CUSTOM_OUTER
    )
    responses["custom"] = {
        "label": "власний плавний НЧ",
        "response": custom_response,
    }

    response_metadata = {}
    for name, item in responses.items():
        response_metadata[name] = response_stats(
            item["response"], radius_full, CUTOFF
        )
    figure_filter_responses(responses, desired_full, radius_full, CUTOFF)

    # фільтруємо зображення в частотній області
    filtered = {}
    input_metrics = evaluate_filter(image, image)
    for name in ("fsamp2", "fwind1", "fwind2", "custom"):
        output_float = fourier.frequency_filter(
            image,
            responses[name]["response"],
            pad_shape=pad_shape,
            pad_mode="reflect",
        )
        output = fourier.to_uint8(output_float)
        path = save_result_image(f"filtered_{name}", output)
        metrics = evaluate_filter(image, output)
        add_energy_ratios(metrics, input_metrics)
        filtered[name] = {
            "label": responses[name]["label"],
            "type": "linear frequency-domain",
            "output": output,
            "image": path,
            "metrics": metrics,
        }

    # порівнюємо доповнення нулями і відбиттям країв
    zero_output = fourier.to_uint8(
        fourier.frequency_filter(
            image,
            custom_response,
            pad_shape=pad_shape,
            pad_mode="zero",
        )
    )
    zero_metrics = evaluate_filter(image, zero_output)
    add_energy_ratios(zero_metrics, input_metrics)
    reflect_metrics = filtered["custom"]["metrics"]
    padding_info = {
        "zero": {
            "psnr": zero_metrics["psnr"],
            "ssim": zero_metrics["ssim"],
        },
        "reflect": {
            "psnr": reflect_metrics["psnr"],
            "ssim": reflect_metrics["ssim"],
        },
    }

    # найкращі фільтри за psnr і ssim
    best_psnr_name = max(
        filtered, key=lambda name: filtered[name]["metrics"]["psnr"]
    )
    best_ssim_name = max(
        filtered, key=lambda name: filtered[name]["metrics"]["ssim"]
    )
    comparison = {
        "best_psnr": best_psnr_name,
        "best_psnr_value": filtered[best_psnr_name]["metrics"]["psnr"],
        "best_ssim": best_ssim_name,
        "best_ssim_value": filtered[best_ssim_name]["metrics"]["ssim"],
    }
    figure_filtered_images(image, filtered)

    # додаємо гауссів шум
    rng = np.random.default_rng(SEED)
    noisy = add_gaussian_noise(image, NOISE_SIGMA, rng)
    noisy_path = save_result_image("noisy_gaussian", noisy)
    noisy_metrics = evaluate_filter(image, noisy)
    add_energy_ratios(noisy_metrics, input_metrics)

    # лінійний нч-фільтр для шуму
    noise_response, _ = fourier.custom_lowpass_response(
        pad_shape, NOISE_INNER, NOISE_OUTER
    )
    linear_output = fourier.to_uint8(
        fourier.frequency_filter(
            noisy,
            noise_response,
            pad_shape=pad_shape,
            pad_mode="reflect",
        )
    )
    linear_path = save_result_image("denoised_linear_freq", linear_output)

    # медіанний і білатеральний фільтри
    median_output = ndimage.median_filter(noisy, size=5, mode="nearest")
    median_path = save_result_image("denoised_median", median_output)
    bilateral_output = bilateral_filter(noisy)
    bilateral_path = save_result_image(
        "denoised_bilateral", bilateral_output
    )

    noise_filters = {}
    definitions = [
        (
            "linear_frequency",
            "linear",
            "лінійний частотний НЧ",
            {
                "inner_cutoff": NOISE_INNER,
                "outer_cutoff": NOISE_OUTER,
                "padding": "reflect",
            },
            linear_output,
            linear_path,
        ),
        (
            "median",
            "nonlinear",
            "медіанний 5×5",
            {"size": 5, "boundary": "nearest"},
            median_output,
            median_path,
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
            bilateral_output,
            bilateral_path,
        ),
    ]
    for key, kind, label, params, output, path in definitions:
        metrics = evaluate_filter(image, output)
        add_energy_ratios(metrics, input_metrics)
        metrics["psnr_gain_vs_noisy"] = (
            metrics["psnr"] - noisy_metrics["psnr"]
        )
        metrics["ssim_gain_vs_noisy"] = (
            metrics["ssim"] - noisy_metrics["ssim"]
        )
        noise_filters[key] = {
            "type": kind,
            "label": label,
            "parameters": params,
            "image": path,
            "metrics_vs_clean": metrics,
        }

    best_noise_psnr = max(
        noise_filters,
        key=lambda name: noise_filters[name]["metrics_vs_clean"]["psnr"],
    )
    best_noise_ssim = max(
        noise_filters,
        key=lambda name: noise_filters[name]["metrics_vs_clean"]["ssim"],
    )
    noise_result = {
        "seed": SEED,
        "sigma": NOISE_SIGMA,
        "noise_model": "адитивний гауссів шум, обрізання до 0...255",
        "noisy": {
            "image": noisy_path,
            "metrics_vs_clean": noisy_metrics,
        },
        "filters": noise_filters,
        "best": {
            "psnr": best_noise_psnr,
            "psnr_value": noise_filters[best_noise_psnr][
                "metrics_vs_clean"
            ]["psnr"],
            "ssim": best_noise_ssim,
            "ssim_value": noise_filters[best_noise_ssim][
                "metrics_vs_clean"
            ]["ssim"],
        },
    }
    figure_noise_comparison(image, noise_result)
    figure_metrics_summary(noise_result)
    figure_radial_spectra(image, filtered, noise_result)

    # збираємо результати у json
    results = {
        "title": "Практична робота 5. Перетворення Фур'є для обробки "
        "зображення",
        "parameters": {
            "seed": SEED,
            "noise_sigma": NOISE_SIGMA,
            "design_size": DESIGN_SIZE,
            "cutoff": CUTOFF,
            "custom_inner": CUSTOM_INNER,
            "custom_outer": CUSTOM_OUTER,
            "noise_inner": NOISE_INNER,
            "noise_outer": NOISE_OUTER,
            "padding": "reflect",
            "padded_shape": [int(pad_shape[0]), int(pad_shape[1])],
        },
        "input": {
            "path": "data/input.png",
            "stats": image_stats(image),
        },
        "fft": {
            "dc_value": dc_value,
            "pixel_sum": pixel_sum,
            "roundtrip_max_abs_error": roundtrip_error,
            "log_magnitude_min": float(log_magnitude.min()),
            "log_magnitude_max": float(log_magnitude.max()),
            "spectrum_image": "out/images/spectrum_log_magnitude.png",
        },
        "freqspace": {
            "even_example": even_example.tolist(),
            "odd_example": odd_example.tolist(),
            "whole_example": whole_example.tolist(),
        },
        "kernels": {
            name: {
                "shape": [
                    int(kernels[name].shape[0]),
                    int(kernels[name].shape[1]),
                ],
                "sum": float(kernels[name].sum()),
                "min": float(kernels[name].min()),
                "max": float(kernels[name].max()),
                "window": window_name,
                "response": response_metadata[name],
            }
            for name, window_name in (
                ("fsamp2", "прямокутне (не застосовується)"),
                ("fwind1", "хеммінга 1D ⊗ хеммінга 1D"),
                ("fwind2", "радіальне ханна (несепарабельне)"),
            )
        },
        "custom_filter": {
            "type": "круговий нч-фільтр із косинусним переходом",
            "inner": CUSTOM_INNER,
            "outer": CUSTOM_OUTER,
            "response": response_metadata["custom"],
            "output_image": filtered["custom"]["image"],
            "metrics_vs_input": filtered["custom"]["metrics"],
        },
        "filters": {
            name: {
                "label": filtered[name]["label"],
                "type": filtered[name]["type"],
                "output_image": filtered[name]["image"],
                "metrics_vs_input": filtered[name]["metrics"],
                "response": response_metadata[name],
            }
            for name in ("fsamp2", "fwind1", "fwind2", "custom")
        },
        "padding_effect": padding_info,
        "noise_experiment": noise_result,
        "outputs": {},
    }
    results["outputs"] = build_output_manifest()
    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(
            json_ready(results),
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")

    print("готово, результати в out/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
