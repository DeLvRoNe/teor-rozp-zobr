# практична робота 7. оброблення кольорових зображень
# запуск: python main.py

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.color import hsv2rgb, lab2rgb, rgb2hsv, rgb2lab
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.measure import shannon_entropy

from utils import color_ssim, image_axis, load_rgb, mse, psnr, save_figure, save_rgb

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"

IMAGE_NAME = "input.png"
CHANNEL_NAMES = ("R", "G", "B")
HISTEQ_BINS = 256
LOCAL_WINDOW = 9
HUE_SATURATION_THRESHOLD = 0.15
CLAHE_TILE_COUNT = 8
CLAHE_CLIP_LIMIT = 0.02
CLAHE_BINS = 256
LAB_PERCENTILES = (1.0, 99.0)
LAB_IMADJUST_GAMMA = 1.0
STATS_DIGITS = 6

# параметри imadjust для каналу v з методички
TASK2_IMADJUST = {
    "low_in": 0.02,
    "high_in": 0.68,
    "gamma": 0.7,
    "low_out": 0.0,
    "high_out": 1.0,
}

# ваги яскравості bt.709
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def setup_console():
    # щоб кирилиця нормально виводилась у консолі
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def relative_path(path):
    # шлях відносно папки практиchna-07
    return path.resolve().relative_to(BASE_DIR).as_posix()


def rounded(value, digits=STATS_DIGITS):
    # округлення для json
    return round(float(value), digits)


def to_unit(image):
    # uint8 -> [0, 1]
    return np.asarray(image, dtype=np.float64) / 255.0


def to_uint8(image):
    # [0, 1] -> uint8
    scaled = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0) * 255.0
    return np.rint(scaled).astype(np.uint8)


def imadjust(channel, low_in, high_in, gamma=1.0, low_out=0.0, high_out=1.0):
    # обрізаємо, нормуємо і підносимо до степеня gamma
    if not high_in > low_in:
        raise ValueError("потрібно high_in > low_in")
    if not high_out > low_out:
        raise ValueError("потрібно high_out > low_out")
    if gamma <= 0.0:
        raise ValueError("gamma має бути додатним")

    data = np.asarray(channel, dtype=np.float64)
    clipped = np.clip(data, low_in, high_in)
    normalized = (clipped - low_in) / (high_in - low_in)
    corrected = np.power(normalized, gamma)
    return low_out + corrected * (high_out - low_out)


def luminance(rgb_unit):
    # яскравість за bt.709
    red, green, blue = LUMINANCE_WEIGHTS
    return (
        red * rgb_unit[..., 0]
        + green * rgb_unit[..., 1]
        + blue * rgb_unit[..., 2]
    )


def channel_entropy_bits(channel_unit):
    # ентропія шеннона каналу з 256 рівнів
    levels = np.clip(np.rint(channel_unit * 255.0), 0, 255).astype(np.uint8)
    return float(shannon_entropy(levels))


def entropy_bits(rgb_unit):
    # ентропія шеннона яскравості
    return channel_entropy_bits(luminance(rgb_unit))


def local_std_map(rgb_unit, window=LOCAL_WINDOW):
    # карта локального середньоквадратичного відхилення
    gray = luminance(rgb_unit)
    mean = uniform_filter(gray, size=window, mode="reflect")
    mean_square = uniform_filter(gray * gray, size=window, mode="reflect")
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return np.sqrt(variance) * 255.0


def local_contrast(rgb_unit, window=LOCAL_WINDOW):
    # середній локальний контраст в рівнях 0-255
    return float(local_std_map(rgb_unit, window).mean())


def colorfulness(rgb_uint8):
    # насиченість кольору за hasler-susstrunk
    data = np.asarray(rgb_uint8, dtype=np.float64)
    red, green, blue = data[..., 0], data[..., 1], data[..., 2]
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    return float(
        np.hypot(red_green.std(), yellow_blue.std())
        + 0.3 * np.hypot(red_green.mean(), yellow_blue.mean())
    )


def image_statistics(image):
    # основні статистики зображення
    unit = to_unit(image)
    gray = luminance(unit)
    return {
        "mean_levels": rounded(image.mean()),
        "std_levels": rounded(image.std()),
        "entropy_bits": rounded(entropy_bits(unit)),
        "luminance_mean": rounded(gray.mean()),
        "luminance_std": rounded(gray.std()),
        "local_contrast_levels": rounded(local_contrast(unit)),
        "colorfulness": rounded(colorfulness(image)),
    }


def comparison_metrics(reference, test):
    # наскільки результат відрізняється від вхідного зображення
    error = mse(reference, test)
    value = psnr(reference, test)
    return {
        "mse": rounded(error),
        "psnr_db": rounded(value) if np.isfinite(value) else None,
        "ssim": rounded(color_ssim(reference, test)),
    }


def hue_difference_degrees(reference_unit, test_unit):
    # зсув колірного тону в градусах для насичених пікселів
    reference_hsv = rgb2hsv(reference_unit)
    test_hsv = rgb2hsv(test_unit)
    difference = np.abs(reference_hsv[..., 0] - test_hsv[..., 0])
    difference = np.minimum(difference, 1.0 - difference) * 360.0
    mask = (reference_hsv[..., 1] >= HUE_SATURATION_THRESHOLD) & (
        test_hsv[..., 1] >= HUE_SATURATION_THRESHOLD
    )
    if not np.any(mask):
        return {"mean_deg": 0.0, "max_deg": 0.0, "pixels": 0}
    values = difference[mask]
    return {
        "mean_deg": rounded(values.mean()),
        "max_deg": rounded(values.max()),
        "pixels": int(values.size),
    }


def per_channel_histeq(rgb_unit):
    # вирівнюємо гістограму кожного каналу r, g, b окремо
    result = rgb_unit.copy()
    for index in range(3):
        result[..., index] = equalize_hist(
            rgb_unit[..., index], nbins=HISTEQ_BINS
        )
    return np.clip(result, 0.0, 1.0)


def hsv_value_histeq(rgb_unit):
    # вирівнюємо гістограму лише каналу v у hsv
    hsv = rgb2hsv(rgb_unit)
    hsv[..., 2] = equalize_hist(hsv[..., 2], nbins=HISTEQ_BINS)
    return np.clip(hsv2rgb(hsv), 0.0, 1.0)


def hsv_contrast(rgb_unit, parameters):
    # підвищуємо контрастність каналу v у hsv
    hsv = rgb2hsv(rgb_unit)
    value_before = hsv[..., 2].copy()
    hsv[..., 2] = imadjust(
        hsv[..., 2],
        low_in=parameters["low_in"],
        high_in=parameters["high_in"],
        gamma=parameters["gamma"],
        low_out=parameters["low_out"],
        high_out=parameters["high_out"],
    )
    value_after = hsv[..., 2]
    result = np.clip(hsv2rgb(hsv), 0.0, 1.0)
    info = {
        "value_before": value_before,
        "value_after": value_after,
        "clipped_low_fraction": float(
            np.mean(value_before < parameters["low_in"])
        ),
        "clipped_high_fraction": float(
            np.mean(value_before > parameters["high_in"])
        ),
    }
    return result, info


def lab_contrast(rgb_unit, method, l_parameters):
    # підвищуємо контрастність каналу l* у lab
    lab = rgb2lab(rgb_unit)
    lightness = lab[..., 0] / 100.0

    if method == "imadjust":
        processed = imadjust(
            lightness,
            low_in=l_parameters["imadjust"]["low_in"],
            high_in=l_parameters["imadjust"]["high_in"],
            gamma=l_parameters["imadjust"]["gamma"],
            low_out=l_parameters["imadjust"]["low_out"],
            high_out=l_parameters["imadjust"]["high_out"],
        )
    elif method == "histeq":
        processed = equalize_hist(lightness, nbins=HISTEQ_BINS)
    elif method == "clahe":
        processed = equalize_adapthist(
            lightness,
            kernel_size=tuple(l_parameters["clahe"]["kernel_size"]),
            clip_limit=l_parameters["clahe"]["clip_limit"],
            nbins=CLAHE_BINS,
        )
    else:
        raise ValueError(f"невідомий метод обробки l*: {method}")

    processed = np.clip(processed, 0.0, 1.0)
    lab_result = lab.copy()
    lab_result[..., 0] = processed * 100.0
    # skimage попереджає про пікселі поза гамутом srgb, вони обрізаються
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = lab2rgb(lab_result)
    return np.clip(result, 0.0, 1.0), processed


def save_output(name, rgb_unit, output):
    # зберігаємо оброблене зображення у png
    image = to_uint8(rgb_unit)
    path = output / f"{name}.png"
    save_rgb(path, image)
    return image, path


def histogram_axis_local(axis, values, title, color, value_range, bins):
    # гістограма одного каналу
    axis.hist(
        values.ravel(),
        bins=bins,
        range=value_range,
        color=color,
        alpha=0.88,
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Значення каналу", fontsize=9)
    axis.set_ylabel("Кількість пікселів", fontsize=9)
    axis.grid(alpha=0.2)


def histogram_overlay_axis(axis, before, after, title, color, value_range, bins=128):
    # накладені гістограми до і після обробки
    axis.hist(
        before.ravel(),
        bins=bins,
        range=value_range,
        histtype="step",
        color="0.45",
        linewidth=1.1,
        label="До обробки",
    )
    axis.hist(
        after.ravel(),
        bins=bins,
        range=value_range,
        histtype="step",
        color=color,
        linewidth=1.5,
        label="Після обробки",
    )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Значення каналу", fontsize=9)
    axis.set_ylabel("Кількість пікселів", fontsize=9)
    axis.legend(fontsize=8, frameon=False)
    axis.grid(alpha=0.2)


def make_task1_figures(
    input_unit,
    input_image,
    task1_unit,
    hsv_unit,
    task1_image,
    hsv_image,
    hue_shift,
    output,
):
    # рисунки до завдання 1
    figures = {}

    path = output / "fig01_task1_result.png"
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 6.4))
    image_axis(axes[0], input_unit, "Оригінал")
    image_axis(axes[1], task1_unit, "Поканальне histeq R, G, B")
    image_axis(axes[2], hsv_unit, "Histeq каналу V у HSV")
    figure.suptitle(
        "П.1. Розтягування гістограми кольорового зображення",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task1_result"] = path

    path = output / "fig02_task1_channels.png"
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 10.2))
    for index, name in enumerate(CHANNEL_NAMES):
        image_axis(
            axes[0, index],
            input_image[..., index],
            f"Канал {name}: оригінал",
            vmin=0,
            vmax=255,
        )
        image_axis(
            axes[1, index],
            task1_image[..., index],
            f"Канал {name}: після histeq",
            vmin=0,
            vmax=255,
        )
    figure.suptitle(
        "П.1. Поканальні зображення до і після вирівнювання гістограм",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task1_channels"] = path

    path = output / "fig03_task1_histograms.png"
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.2))
    channel_colors = ("tab:red", "tab:green", "tab:blue")
    for index, name in enumerate(CHANNEL_NAMES):
        histogram_axis_local(
            axes[0, index],
            input_image[..., index],
            f"Гістограма {name}: оригінал",
            channel_colors[index],
            (0.0, 255.0),
            HISTEQ_BINS,
        )
        histogram_axis_local(
            axes[1, index],
            task1_image[..., index],
            f"Гістограма {name}: після histeq",
            channel_colors[index],
            (0.0, 255.0),
            HISTEQ_BINS,
        )
    figure.suptitle(
        "П.1. Гістограми каналів R, G, B до і після поканального histeq",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task1_histograms"] = path

    path = output / "fig04_task1_color_shift.png"
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.4))
    input_hsv = rgb2hsv(input_unit)
    result_hsv = rgb2hsv(task1_image / 255.0)
    image_axis(
        axes[0],
        input_hsv[..., 1],
        "Насиченість S: оригінал",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis(
        axes[1],
        result_hsv[..., 1],
        "Насиченість S: після histeq",
        vmin=0.0,
        vmax=1.0,
    )
    hue_difference = np.abs(input_hsv[..., 0] - result_hsv[..., 0])
    hue_difference = np.minimum(hue_difference, 1.0 - hue_difference) * 360.0
    mask = (input_hsv[..., 1] >= HUE_SATURATION_THRESHOLD) & (
        result_hsv[..., 1] >= HUE_SATURATION_THRESHOLD
    )
    hue_map = np.where(mask, hue_difference, np.nan)
    image = axes[2].imshow(
        hue_map,
        cmap="magma",
        vmin=0.0,
        vmax=15.0,
        interpolation="nearest",
    )
    axes[2].set_title(
        f"Зсув тону |ΔH|, градуси\n"
        f"середній {hue_shift['mean_deg']:.2f}, "
        f"максимум {hue_shift['max_deg']:.2f}",
        fontsize=10,
    )
    axes[2].axis("off")
    colorbar = figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.02)
    colorbar.set_label("ΔH, градуси", fontsize=8)
    figure.suptitle(
        "П.1. Зміна насиченості та колірного тону після поканального histeq",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task1_color_shift"] = path

    return figures


def make_task2_figures(input_unit, task2_unit, task2_hsv, output):
    # рисунки до завдання 2
    figures = {}

    input_hsv = rgb2hsv(input_unit)
    result_hsv = rgb2hsv(task2_unit)

    path = output / "fig05_task2_channels.png"
    figure, axes = plt.subplots(2, 4, figsize=(15.5, 8.2))
    image_axis(axes[0, 0], input_unit, "Оригінал: RGB")
    image_axis(
        axes[0, 1],
        input_hsv[..., 0],
        f"H: колірний тон\nсереднє {input_hsv[..., 0].mean():.3f}",
        cmap="hsv",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis(
        axes[0, 2],
        input_hsv[..., 1],
        f"S: насиченість\nсереднє {input_hsv[..., 1].mean():.3f}",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis(
        axes[0, 3],
        input_hsv[..., 2],
        f"V: яскравість\nсереднє {input_hsv[..., 2].mean():.3f}",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis(axes[1, 0], task2_unit, "Результат: HSV + imadjust(V)")
    image_axis(
        axes[1, 1],
        result_hsv[..., 0],
        f"H: без змін\nсереднє {result_hsv[..., 0].mean():.3f}",
        cmap="hsv",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis(
        axes[1, 2],
        result_hsv[..., 1],
        f"S: без змін\nсереднє {result_hsv[..., 1].mean():.3f}",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis(
        axes[1, 3],
        result_hsv[..., 2],
        f"V: після imadjust\nсереднє {result_hsv[..., 2].mean():.3f}",
        vmin=0.0,
        vmax=1.0,
    )
    figure.suptitle(
        "П.2. Канали H, S, V до і після контрастування складової яскравості",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task2_channels"] = path

    path = output / "fig06_task2_histograms.png"
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    histogram_overlay_axis(
        axes[0],
        input_hsv[..., 0],
        result_hsv[..., 0],
        "Гістограма H: тон",
        "tab:purple",
        (0.0, 1.0),
    )
    histogram_overlay_axis(
        axes[1],
        input_hsv[..., 1],
        result_hsv[..., 1],
        "Гістограма S: насиченість",
        "tab:green",
        (0.0, 1.0),
    )
    histogram_overlay_axis(
        axes[2],
        task2_hsv["value_before"],
        task2_hsv["value_after"],
        "Гістограма V: яскравість",
        "tab:red",
        (0.0, 1.0),
    )
    figure.suptitle(
        "П.2. Гістограми каналів HSV до і після imadjust каналу V",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task2_histograms"] = path

    path = output / "fig07_task2_result.png"
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8))
    image_axis(axes[0], input_unit, "Оригінал")
    image_axis(
        axes[1],
        task2_unit,
        "HSV, imadjust(V, [0.02; 0.68], gamma = 0.7)",
    )
    figure.suptitle(
        "П.2. Підвищення контрастності темного зображення у просторі HSV",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task2_result"] = path

    return figures


def make_task3_figures(input_unit, input_lab, results, l_channels, output):
    # рисунки до завдання 3
    figures = {}

    path = output / "fig08_task3_lab_channels.png"
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 5.6))
    image_axis(axes[0], input_unit, "Оригінал: RGB")
    image_axis(
        axes[1],
        input_lab[..., 0],
        f"L*: світлота [0; 100]\n"
        f"середнє {input_lab[..., 0].mean():.2f}",
        vmin=0.0,
        vmax=100.0,
    )
    image_axis(
        axes[2],
        input_lab[..., 1],
        f"a*: зелений-червоний\n"
        f"діапазон {input_lab[..., 1].min():.1f}...{input_lab[..., 1].max():.1f}",
        cmap="coolwarm",
        vmin=-110.0,
        vmax=110.0,
    )
    image_axis(
        axes[3],
        input_lab[..., 2],
        f"b*: синій-жовтий\n"
        f"діапазон {input_lab[..., 2].min():.1f}...{input_lab[..., 2].max():.1f}",
        cmap="coolwarm",
        vmin=-110.0,
        vmax=110.0,
    )
    figure.suptitle(
        "П.3. Подання зображення у просторі L*a*b*: канали L*, a*, b*",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task3_channels"] = path

    path = output / "fig09_task3_results.png"
    figure, axes = plt.subplots(1, 4, figsize=(16.5, 5.8))
    image_axis(axes[0], input_unit, "Оригінал")
    image_axis(axes[1], results["imadjust"], "L*: imadjust (1...99 %)")
    image_axis(axes[2], results["histeq"], "L*: histeq (256 бін)")
    image_axis(
        axes[3],
        results["clahe"],
        "L*: adapthisteq (CLAHE)\nclip_limit = 0.02",
    )
    figure.suptitle(
        "П.3. Підвищення контрастності каналу L* у просторі L*a*b*",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task3_results"] = path

    path = output / "fig10_task3_l_histograms.png"
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    methods = (
        ("imadjust", "imadjust", "tab:orange"),
        ("histeq", "histeq", "tab:blue"),
        ("clahe", "CLAHE", "tab:green"),
    )
    for axis, (key, label, color) in zip(axes, methods):
        histogram_overlay_axis(
            axis,
            input_lab[..., 0] / 100.0,
            l_channels[key],
            f"L*: {label}\n"
            f"середнє {l_channels[key].mean() * 100.0:.2f}, "
            f"СКВ {l_channels[key].std() * 100.0:.2f}",
            color,
            (0.0, 1.0),
            bins=100,
        )
    figure.suptitle(
        "П.3. Гістограми каналу L* до і після трьох методів контрастування",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task3_l_histograms"] = path

    path = output / "fig11_task3_local_contrast.png"
    figure, axes = plt.subplots(1, 4, figsize=(16.0, 5.2))
    input_map = local_std_map(input_unit)
    clahe_map = local_std_map(results["clahe"])
    upper = max(float(np.percentile(input_map, 99.5)), float(np.percentile(clahe_map, 99.5)))
    image_axis(axes[0], luminance(input_unit), "Яскравість: оригінал", vmin=0.0, vmax=1.0)
    image_axis(
        axes[1],
        input_map,
        f"Локальний контраст: оригінал\nсередній {input_map.mean():.2f}",
        vmin=0.0,
        vmax=upper,
    )
    image_axis(
        axes[2],
        clahe_map,
        f"Локальний контраст: CLAHE\nсередній {clahe_map.mean():.2f}",
        vmin=0.0,
        vmax=upper,
    )
    difference = clahe_map - input_map
    limit = max(float(np.percentile(np.abs(difference), 99.5)), 1.0)
    image_axis(
        axes[3],
        difference,
        f"Приріст локального контрасту\n"
        f"максимум {difference.max():.2f}",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    figure.suptitle(
        f"П.3. Локальний контраст (вікно {LOCAL_WINDOW}x{LOCAL_WINDOW} пікселів)",
        fontsize=12,
    )
    save_figure(figure, path)
    figures["task3_local_contrast"] = path

    return figures


def run(data_path, output):
    print("практична робота 7. оброблення кольорових зображень")

    if not data_path.exists():
        raise FileNotFoundError(f"не знайдено вхідне зображення: {data_path}")

    # читаємо картинку
    input_image = load_rgb(data_path)
    if input_image.ndim != 3 or input_image.shape[2] != 3:
        raise ValueError("очікується кольорове зображення rgb")
    height, width = input_image.shape[:2]
    input_unit = to_unit(input_image)
    input_stats = image_statistics(input_image)

    channel_means = {}
    for index, name in enumerate(CHANNEL_NAMES):
        channel_means[name] = rounded(input_image[..., index].mean())

    print(f"вхідне зображення: {relative_path(data_path)}")
    print(f"розмір: {height} x {width}, канали: 3, клас: uint8")
    print(
        f"яскравість: середнє {input_stats['mean_levels']:.2f} рівня, "
        f"скв {input_stats['std_levels']:.2f} рівня"
    )
    print(
        "середні каналів r/g/b: "
        f"{channel_means['R']:.2f} / {channel_means['G']:.2f} / {channel_means['B']:.2f}"
    )
    print(
        f"ентропія яскравості: {input_stats['entropy_bits']:.3f} біт, "
        f"локальний контраст: {input_stats['local_contrast_levels']:.2f} рівня"
    )

    # завдання 1. поканальне вирівнювання гістограм
    print("\nзавдання 1. розтягування гістограми кольорового зображення")
    task1_unit = per_channel_histeq(input_unit)
    hsv_unit = hsv_value_histeq(input_unit)
    task1_image, task1_path = save_output("task1_per_channel_histeq", task1_unit, output)
    hsv_image, hsv_path = save_output("task1_hsv_v_histeq", hsv_unit, output)
    task1_stats = image_statistics(task1_image)
    hsv_stats = image_statistics(hsv_image)
    hue_shift_channels = hue_difference_degrees(input_unit, task1_image / 255.0)
    hue_shift_hsv = hue_difference_degrees(input_unit, hsv_image / 255.0)
    input_hsv = rgb2hsv(input_unit)
    spread_input = float(np.ptp(list(channel_means.values())))
    spread_result = float(
        np.ptp([rounded(task1_image[..., i].mean()) for i in range(3)])
    )
    print(
        f"поканальне histeq r, g, b ({HISTEQ_BINS} бін): скв "
        f"{task1_stats['std_levels']:.2f} рівня, ентропія "
        f"{task1_stats['entropy_bits']:.3f} біт, локальний контраст "
        f"{task1_stats['local_contrast_levels']:.2f} рівня"
    )
    print(
        f"histeq каналу v у hsv: скв {hsv_stats['std_levels']:.2f} рівня, "
        f"ентропія {hsv_stats['entropy_bits']:.3f} біт"
    )
    print(
        f"зсув тону (поканально / hsv-v): "
        f"{hue_shift_channels['mean_deg']:.3f} / {hue_shift_hsv['mean_deg']:.3f} градусів"
    )
    print(f"збережено: {relative_path(task1_path)}")
    print(f"збережено: {relative_path(hsv_path)}")

    task1_figures = make_task1_figures(
        input_unit,
        input_image,
        task1_unit,
        hsv_unit,
        task1_image,
        hsv_image,
        hue_shift_channels,
        output,
    )
    for path in task1_figures.values():
        print(f"рисунок: {relative_path(path)}")

    # завдання 2. hsv і imadjust каналу v
    print("\nзавдання 2. перетворення rgb -> hsv і підвищення контрастності")
    task2_unit, hsv_info = hsv_contrast(input_unit, TASK2_IMADJUST)
    task2_image, task2_path = save_output("task2_hsv_imadjust", task2_unit, output)
    task2_stats = image_statistics(task2_image)
    task2_hsv = rgb2hsv(task2_image / 255.0)
    task2_hue_shift = hue_difference_degrees(input_unit, task2_image / 255.0)
    print(
        f"параметри imadjust: low_in={TASK2_IMADJUST['low_in']}, "
        f"high_in={TASK2_IMADJUST['high_in']}, gamma={TASK2_IMADJUST['gamma']}"
    )
    print(
        f"яскравість: середнє {input_stats['mean_levels']:.2f} -> "
        f"{task2_stats['mean_levels']:.2f} рівня, скв "
        f"{input_stats['std_levels']:.2f} -> {task2_stats['std_levels']:.2f} рівня"
    )
    print(
        f"локальний контраст: {input_stats['local_contrast_levels']:.2f} -> "
        f"{task2_stats['local_contrast_levels']:.2f} рівня"
    )
    print(
        f"насичення до 1.0: {hsv_info['clipped_high_fraction'] * 100.0:.2f} % пікселів"
    )
    print(f"зсув тону: {task2_hue_shift['mean_deg']:.4f} градусів")
    print(f"збережено: {relative_path(task2_path)}")

    task2_figures = make_task2_figures(input_unit, task2_unit, hsv_info, output)
    for path in task2_figures.values():
        print(f"рисунок: {relative_path(path)}")

    # завдання 3. lab і контрастування каналу l*
    print("\nзавдання 3. перетворення rgb -> lab і підвищення контрастності")
    input_lab = rgb2lab(input_unit)
    lightness = input_lab[..., 0] / 100.0
    low_in, high_in = (float(value) for value in np.percentile(lightness, LAB_PERCENTILES))
    kernel_size = [
        max(8, height // CLAHE_TILE_COUNT),
        max(8, width // CLAHE_TILE_COUNT),
    ]
    lab_parameters = {
        "imadjust": {
            "low_in": low_in,
            "high_in": high_in,
            "gamma": LAB_IMADJUST_GAMMA,
            "low_out": 0.0,
            "high_out": 1.0,
            "percentiles": list(LAB_PERCENTILES),
        },
        "histeq": {"bins": HISTEQ_BINS},
        "clahe": {
            "kernel_size": kernel_size,
            "clip_limit": CLAHE_CLIP_LIMIT,
            "bins": CLAHE_BINS,
        },
    }
    print(
        f"канал l*: середнє {input_lab[..., 0].mean():.2f}, "
        f"скв {input_lab[..., 0].std():.2f}, "
        f"1-й/99-й процентилі {low_in:.4f} / {high_in:.4f}"
    )
    print(
        f"imadjust(l*): gamma={LAB_IMADJUST_GAMMA}, "
        f"межі [{low_in:.4f}; {high_in:.4f}]"
    )
    print(f"histeq(l*): {HISTEQ_BINS} бін")
    print(
        f"adapthisteq(l*): вікно {kernel_size[0]}x{kernel_size[1]}, "
        f"clip_limit={CLAHE_CLIP_LIMIT}, {CLAHE_BINS} бін"
    )

    lab_units = {}
    lab_images = {}
    lab_channels = {}
    lab_paths = {}
    for method in ("imadjust", "histeq", "clahe"):
        result_unit, processed_l = lab_contrast(input_unit, method, lab_parameters)
        result_image, path = save_output(f"task3_lab_{method}", result_unit, output)
        lab_units[method] = result_unit
        lab_images[method] = result_image
        lab_channels[method] = processed_l
        lab_paths[method] = path
        stats = image_statistics(result_image)
        print(
            f"{method}: скв {stats['std_levels']:.2f} рівня, ентропія "
            f"{stats['entropy_bits']:.3f} біт, локальний контраст "
            f"{stats['local_contrast_levels']:.2f} рівня, "
            f"l* середнє {processed_l.mean() * 100.0:.2f}"
        )
        print(f"  збережено: {relative_path(path)}")

    task3_figures = make_task3_figures(
        input_unit, input_lab, lab_units, lab_channels, output
    )
    for path in task3_figures.values():
        print(f"рисунок: {relative_path(path)}")

    # порівняння трьох методів для каналу l*
    method_names = {
        "imadjust": "imadjust",
        "histeq": "histeq",
        "clahe": "CLAHE",
    }
    method_comparison = []
    for method in ("imadjust", "histeq", "clahe"):
        stats = image_statistics(lab_images[method])
        method_comparison.append(
            {
                "method": method_names[method],
                "l_mean": rounded(lab_channels[method].mean() * 100.0),
                "l_std": rounded(lab_channels[method].std() * 100.0),
                "std_levels": stats["std_levels"],
                "entropy_bits": stats["entropy_bits"],
                "local_contrast_levels": stats["local_contrast_levels"],
                "ssim": rounded(color_ssim(input_image, lab_images[method])),
            }
        )

    # збираємо результати
    images = {
        "per_channel_histeq": (task1_image, task1_path),
        "hsv_v_histeq": (hsv_image, hsv_path),
        "hsv_contrast": (task2_image, task2_path),
        "lab_imadjust": (lab_images["imadjust"], lab_paths["imadjust"]),
        "lab_histeq": (lab_images["histeq"], lab_paths["histeq"]),
        "lab_clahe": (lab_images["clahe"], lab_paths["clahe"]),
    }
    image_entries = {}
    for name, (image, path) in images.items():
        image_entries[name] = {
            "path": relative_path(path),
            "stats": image_statistics(image),
            "compare": comparison_metrics(input_image, image),
        }

    figures = {}
    for mapping in (task1_figures, task2_figures, task3_figures):
        for key, path in mapping.items():
            figures[key] = relative_path(path)

    results = {
        "work": "praktychna-07",
        "title": "Практична робота 7. Оброблення кольорових зображень",
        "input": {
            "path": relative_path(data_path),
            "height": int(height),
            "width": int(width),
            "channel_means": channel_means,
            "stats": input_stats,
        },
        "parameters": {
            "histeq_bins": HISTEQ_BINS,
            "task2_imadjust": dict(TASK2_IMADJUST),
            "task3": {
                "l_percentiles": list(LAB_PERCENTILES),
                "imadjust": {
                    "low_in": low_in,
                    "high_in": high_in,
                    "gamma": LAB_IMADJUST_GAMMA,
                    "low_out": 0.0,
                    "high_out": 1.0,
                },
                "histeq": {"bins": HISTEQ_BINS},
                "clahe": {
                    "kernel_size": list(kernel_size),
                    "clip_limit": CLAHE_CLIP_LIMIT,
                    "bins": CLAHE_BINS,
                },
            },
            "local_contrast_window": LOCAL_WINDOW,
            "hue_saturation_threshold": HUE_SATURATION_THRESHOLD,
        },
        "task1": {
            "channel_mean_spread_input": rounded(spread_input),
            "channel_mean_spread_result": rounded(spread_result),
            "hue_shift_per_channel": hue_shift_channels,
            "hue_shift_hsv_v": hue_shift_hsv,
            "saturation_mean_input": rounded(input_hsv[..., 1].mean()),
            "saturation_mean_per_channel": rounded(
                rgb2hsv(task1_image / 255.0)[..., 1].mean()
            ),
            "saturation_mean_hsv_v": rounded(
                rgb2hsv(hsv_image / 255.0)[..., 1].mean()
            ),
        },
        "task2": {
            "v_channel": {
                "mean_before": rounded(input_hsv[..., 2].mean()),
                "mean_after": rounded(task2_hsv[..., 2].mean()),
                "std_before": rounded(input_hsv[..., 2].std()),
                "std_after": rounded(task2_hsv[..., 2].std()),
                "p01_before": rounded(np.percentile(input_hsv[..., 2], 1.0)),
                "p99_before": rounded(np.percentile(input_hsv[..., 2], 99.0)),
                "p01_after": rounded(np.percentile(task2_hsv[..., 2], 1.0)),
                "p99_after": rounded(np.percentile(task2_hsv[..., 2], 99.0)),
            },
            "clipped_low_fraction": rounded(hsv_info["clipped_low_fraction"]),
            "clipped_high_fraction": rounded(hsv_info["clipped_high_fraction"]),
            "hue_shift": task2_hue_shift,
            "saturation_mean_before": rounded(input_hsv[..., 1].mean()),
            "saturation_mean_after": rounded(task2_hsv[..., 1].mean()),
            "luminance_mean_before": input_stats["luminance_mean"],
            "luminance_mean_after": task2_stats["luminance_mean"],
            "luminance_std_before": input_stats["luminance_std"],
            "luminance_std_after": task2_stats["luminance_std"],
        },
        "task3": {
            "l_percentiles": {
                "p01": rounded(np.percentile(lightness, 1.0)),
                "p99": rounded(np.percentile(lightness, 99.0)),
            },
            "method_comparison": method_comparison,
        },
        "images": image_entries,
        "figures": figures,
    }

    # записуємо результати у json
    results_path = output / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nрезультати збережено: {relative_path(results_path)}")
    print("роботу завершено")


def main():
    setup_console()
    data_path = DATA_DIR / IMAGE_NAME
    output = OUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    run(data_path, output)


if __name__ == "__main__":
    main()
