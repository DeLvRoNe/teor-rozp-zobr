# практична робота 2: покращення зображення за допомогою еквалізації
# запуск: python main.py

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist

from utils import (
    histogram_axis,
    image_axis,
    load_gray,
    psnr,
    save_figure,
    save_gray,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
INPUT_PATH = DATA_DIR / "input.png"

SEED = 42

# значення gamma
GAMMA_CASES = (1.0, 0.5, 2.1)
# межі вхідного і вихідного діапазону для imadjust
INPUT_RANGE = (0.2, 0.6)
OUTPUT_RANGE = (0.2, 0.6)

# параметри clahe
CLAHE_CLIP_LIMIT = 0.01
CLAHE_NBINS = 256
CLAHE_TILES = 8

# параметри шуму
GAUSSIAN_MEAN = 0.0
GAUSSIAN_VARIANCE = 0.01
SALT_PEPPER_DENSITY = 0.05
SPECKLE_VARIANCE = 0.04

FIGURE_FILES = [
    "01_input_image.png",
    "02_input_histogram.png",
    "03_histeq_result.png",
    "04_imadjust_gamma.png",
    "05_imadjust_transfer_curves.png",
    "06_imadjust_input_range.png",
    "07_imadjust_output_range.png",
    "08_histeq_vs_adapthisteq.png",
    "09_noise_types.png",
]


# точкові перетворення
def to_uint8(values):
    # округлюємо і переводимо в uint8
    array = np.asarray(values, dtype=np.float64)
    return np.clip(np.floor(array + 0.5), 0.0, 255.0).astype(np.uint8)


def histogram_entropy(image):
    # ентропія гістограми в бітах
    counts = np.bincount(
        np.asarray(image, dtype=np.uint8).ravel(), minlength=256
    ).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def image_stats(image):
    # збираємо статистики зображення
    array = np.asarray(image)
    values = array.astype(np.float64)
    percentile_5, percentile_95 = np.percentile(values, [5, 95])
    return {
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(round(values.mean(), 4)),
        "std": float(round(values.std(), 4)),
        "entropy": float(round(histogram_entropy(array), 4)),
        "percentile_5": float(round(percentile_5, 1)),
        "percentile_95": float(round(percentile_95, 1)),
    }


def histeq(image, levels=256):
    # глобальна еквалізація через cdf
    image = np.asarray(image, dtype=np.uint8)
    counts = np.bincount(image.ravel(), minlength=levels).astype(np.float64)
    cdf = np.cumsum(counts)
    cdf_min = cdf[counts > 0][0]
    denominator = float(image.size - cdf_min)
    if denominator <= 0.0:
        return image.copy()
    lut = np.floor((cdf - cdf_min) / denominator * (levels - 1) + 0.5)
    lut = np.clip(lut, 0, levels - 1).astype(np.uint8)
    return lut[image]


def imadjust_transfer(low_in=0.0, high_in=1.0, low_out=0.0, high_out=1.0, gamma=1.0):
    # будуємо передавальну характеристику imadjust
    if not 0.0 <= low_in < high_in <= 1.0:
        raise ValueError("потрібно 0 <= low_in < high_in <= 1")
    if not 0.0 <= low_out < high_out <= 1.0:
        raise ValueError("потрібно 0 <= low_out < high_out <= 1")
    if gamma <= 0.0:
        raise ValueError("gamma має бути додатним")

    source = np.linspace(0.0, 1.0, 256)
    transfer = np.clip(source, low_in, high_in)
    transfer = (transfer - low_in) / (high_in - low_in)
    transfer = np.power(transfer, gamma)
    transfer = low_out + transfer * (high_out - low_out)
    return np.clip(transfer, 0.0, 1.0)


def imadjust(image, low_in=0.0, high_in=1.0, low_out=0.0, high_out=1.0, gamma=1.0):
    # корекція контрасту і яскравості
    transfer = imadjust_transfer(low_in, high_in, low_out, high_out, gamma)
    return to_uint8(transfer[np.asarray(image, dtype=np.uint8)] * 255.0)


# моделі шуму
def add_gaussian_noise(image, rng, mean=GAUSSIAN_MEAN, variance=GAUSSIAN_VARIANCE):
    # адитивний гаусів шум
    normalized = image.astype(np.float64) / 255.0
    noise = rng.normal(loc=mean, scale=np.sqrt(variance), size=image.shape)
    return to_uint8(np.clip(normalized + noise, 0.0, 1.0) * 255.0)


def add_salt_pepper_noise(image, rng, density=SALT_PEPPER_DENSITY):
    # імпульсний шум сіль і перець
    if not 0.0 <= density <= 1.0:
        raise ValueError("щільність має лежати в діапазоні [0, 1]")

    damaged = rng.random(image.shape) < density
    salt = rng.random(image.shape) < 0.5
    result = image.copy()
    result[damaged & salt] = 255
    result[damaged & ~salt] = 0
    return result


def add_poisson_noise(image, rng):
    # пуассонівський шум, залежить від яскравості пікселя
    counts = image.astype(np.float64)
    noisy = rng.poisson(counts).astype(np.float64)
    return to_uint8(np.clip(noisy, 0.0, 255.0))


def add_speckle_noise(image, rng, variance=SPECKLE_VARIANCE):
    # мультиплікативний шум
    normalized = image.astype(np.float64) / 255.0
    spread = np.sqrt(3.0 * variance)
    multiplicative = rng.uniform(-spread, spread, size=image.shape)
    noisy = normalized + multiplicative * normalized
    return to_uint8(np.clip(noisy, 0.0, 1.0) * 255.0)


# рисунки
def save_input_figure(image, path):
    # вхідне зображення
    figure, axis = plt.subplots(figsize=(5.5, 6.5))
    image_axis(
        axis,
        image,
        f"П. 1. Вхідне напівтонове зображення\n"
        f"{image.shape[0]}x{image.shape[1]} (рядки x стовпці), uint8",
    )
    save_figure(figure, path)


def save_input_histogram(image, path):
    # гістограма вхідного зображення
    figure, axis = plt.subplots(figsize=(8.0, 4.5))
    histogram_axis(
        axis,
        image,
        "П. 2. Гістограма вхідного зображення (256 бінів)",
    )
    axis.axvline(
        float(image.mean()),
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"середнє = {image.mean():.1f}",
    )
    axis.legend(fontsize=9)
    save_figure(figure, path)


def save_histeq_figure(image, equalized, stats_in, stats_eq, path):
    # оригінал і результат глобальної еквалізації
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    image_axis(
        axes[0, 0],
        image,
        f"Оригінал\nсереднє {stats_in['mean']:.1f}, std {stats_in['std']:.1f}",
    )
    image_axis(
        axes[0, 1],
        equalized,
        f"Після histeq\nсереднє {stats_eq['mean']:.1f}, std {stats_eq['std']:.1f}",
    )
    histogram_axis(axes[1, 0], image, "Гістограма оригіналу")
    histogram_axis(axes[1, 1], equalized, "Гістограма після histeq")
    figure.suptitle("П. 3.1. Глобальна еквалізація гістограми (histeq)", fontsize=13)
    save_figure(figure, path)


def save_gamma_figure(results, path):
    # вплив gamma на зображення і гістограму
    figure, axes = plt.subplots(3, 2, figsize=(11.0, 13.0))
    for row, (gamma, result, stats) in enumerate(results):
        image_axis(
            axes[row, 0],
            result,
            f"gamma = {gamma:g}: зображення\n"
            f"середнє {stats['mean']:.1f}, std {stats['std']:.1f}",
        )
        histogram_axis(
            axes[row, 1],
            result,
            f"gamma = {gamma:g}: гістограма\n"
            f"діапазон {stats['min']}..{stats['max']}",
        )
    figure.suptitle("П. 3.2, а. imadjust: вплив gamma при [0 1] -> [0 1]", fontsize=13)
    save_figure(figure, path)


def save_transfer_curves(path):
    # передавальні характеристики для всіх випадків imadjust
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    source = np.linspace(0.0, 1.0, 256)

    axis.plot(
        source,
        source,
        color="0.7",
        linestyle=":",
        linewidth=1.0,
        label="тотожне перетворення",
    )
    for gamma, color in zip(GAMMA_CASES, ("tab:blue", "tab:green", "tab:red")):
        transfer = imadjust_transfer(0.0, 1.0, 0.0, 1.0, gamma)
        axis.plot(
            source,
            transfer,
            color=color,
            label=f"gamma = {gamma:g}, [0 1] -> [0 1]",
        )

    transfer_b = imadjust_transfer(INPUT_RANGE[0], INPUT_RANGE[1], 0.0, 1.0, 1.0)
    axis.plot(
        source,
        transfer_b,
        color="tab:orange",
        linestyle="--",
        label="[0.2 0.6] -> [0 1], gamma = 1",
    )
    transfer_c = imadjust_transfer(0.0, 1.0, OUTPUT_RANGE[0], OUTPUT_RANGE[1], 1.0)
    axis.plot(
        source,
        transfer_c,
        color="tab:purple",
        linestyle="--",
        label="[0 1] -> [0.2 0.6], gamma = 1",
    )

    axis.set_title("П. 3.2. Передавальні характеристики imadjust", fontsize=12)
    axis.set_xlabel("Вхідна яскравість (нормована)")
    axis.set_ylabel("Вихідна яскравість (нормована)")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    save_figure(figure, path)


def save_threshold_figure(image, result, stats_in, stats_out, title, path):
    # порівняння оригіналу і результату imadjust
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
    image_axis(
        axes[0, 0],
        image,
        f"Оригінал\nсереднє {stats_in['mean']:.1f}, std {stats_in['std']:.1f}",
    )
    image_axis(
        axes[0, 1],
        result,
        f"Після imadjust\nсереднє {stats_out['mean']:.1f}, std {stats_out['std']:.1f}",
    )
    histogram_axis(axes[1, 0], image, "Гістограма оригіналу")
    histogram_axis(axes[1, 1], result, "Гістограма після imadjust")
    figure.suptitle(title, fontsize=12)
    save_figure(figure, path)


def save_adaptive_comparison(image, equalized, adaptive, path):
    # порівняння histeq та adapthisteq
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    panels = [
        (image, "Оригінал"),
        (equalized, "histeq (глобальна еквалізація)"),
        (adaptive, "adapthisteq (локальна, CLAHE)"),
    ]
    for column, (panel, title) in enumerate(panels):
        image_axis(axes[0, column], panel, title)
        histogram_axis(axes[1, column], panel, f"Гістограма: {title}")
    figure.suptitle("П. 3.3. Порівняння histeq та adapthisteq (CLAHE)", fontsize=13)
    save_figure(figure, path)


def save_noise_figure(cases, path):
    # зашумлені зображення і їх гістограми
    figure, axes = plt.subplots(4, 2, figsize=(11.5, 15.0))
    for row, case in enumerate(cases):
        stats = case["stats"]
        image_axis(
            axes[row, 0],
            case["image"],
            f"{case['title']}\nсереднє {stats['mean']:.1f}, std {stats['std']:.1f}",
        )
        histogram_axis(
            axes[row, 1],
            case["image"],
            f"Гістограма: {case['short_title']}",
        )
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylabel("Кількість пікселів (лог. шкала)")
    figure.suptitle("П. 4. Зашумлені зображення та їх гістограми", fontsize=13)
    save_figure(figure, path)


def configure_console():
    # щоб український текст не псувався у вікні консолі
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main():
    configure_console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # читаємо вхідну картинку
    image = load_gray(INPUT_PATH)
    image_stats_in = image_stats(image)

    print("Практична робота 2: покращення зображення за допомогою еквалізації")
    print(f"Вхідний файл: {INPUT_PATH}")
    print(
        f"Розмір: {image.shape[0]}x{image.shape[1]}, "
        f"діапазон: {image_stats_in['min']}..{image_stats_in['max']}, "
        f"середнє: {image_stats_in['mean']:.2f}, "
        f"std: {image_stats_in['std']:.2f}, "
        f"ентропія: {image_stats_in['entropy']:.2f} біт"
    )

    print("\nП. 1-2. Вхідне зображення та його гістограма")
    save_input_figure(image, OUT_DIR / FIGURE_FILES[0])
    save_input_histogram(image, OUT_DIR / FIGURE_FILES[1])

    print("\nП. 3.1. Глобальна еквалізація (histeq)")
    equalized = histeq(image)
    equalized_stats = image_stats(equalized)
    save_gray(OUT_DIR / "histeq.png", equalized)
    save_histeq_figure(
        image,
        equalized,
        image_stats_in,
        equalized_stats,
        OUT_DIR / FIGURE_FILES[2],
    )
    print(
        f"Після histeq: середнє {equalized_stats['mean']:.2f}, "
        f"std {equalized_stats['std']:.2f}, "
        f"ентропія {equalized_stats['entropy']:.2f} біт"
    )

    print("\nП. 3.2. Корекція контрасту imadjust")
    gamma_results = []
    for gamma in GAMMA_CASES:
        processed = imadjust(image, 0.0, 1.0, 0.0, 1.0, gamma)
        gamma_results.append((gamma, processed, image_stats(processed)))

    case_specs = [
        {
            "id": "a_gamma_1",
            "gamma": 1.0,
            "low_in": 0.0,
            "high_in": 1.0,
            "low_out": 0.0,
            "high_out": 1.0,
            "file": "imadjust_a_gamma_1.png",
        },
        {
            "id": "a_gamma_0_5",
            "gamma": 0.5,
            "low_in": 0.0,
            "high_in": 1.0,
            "low_out": 0.0,
            "high_out": 1.0,
            "file": "imadjust_a_gamma_0_5.png",
        },
        {
            "id": "a_gamma_2_1",
            "gamma": 2.1,
            "low_in": 0.0,
            "high_in": 1.0,
            "low_out": 0.0,
            "high_out": 1.0,
            "file": "imadjust_a_gamma_2_1.png",
        },
        {
            "id": "b",
            "gamma": 1.0,
            "low_in": INPUT_RANGE[0],
            "high_in": INPUT_RANGE[1],
            "low_out": 0.0,
            "high_out": 1.0,
            "file": "imadjust_b.png",
        },
        {
            "id": "c",
            "gamma": 1.0,
            "low_in": 0.0,
            "high_in": 1.0,
            "low_out": OUTPUT_RANGE[0],
            "high_out": OUTPUT_RANGE[1],
            "file": "imadjust_c.png",
        },
    ]

    normalized = image.astype(np.float64) / 255.0
    imadjust_cases = []
    processed_by_id = {}
    for spec in case_specs:
        processed = imadjust(
            image,
            spec["low_in"],
            spec["high_in"],
            spec["low_out"],
            spec["high_out"],
            spec["gamma"],
        )
        processed_by_id[spec["id"]] = processed
        stats = image_stats(processed)
        save_gray(OUT_DIR / spec["file"], processed)

        # рахуємо контраст усередині робочого діапазону
        in_range = (normalized >= spec["low_in"]) & (normalized <= spec["high_in"])
        input_in_range_std = float(image[in_range].std())
        output_in_range_std = float(processed[in_range].std())
        if input_in_range_std > 0.0:
            in_range_std_ratio = output_in_range_std / input_in_range_std
        else:
            in_range_std_ratio = 1.0

        imadjust_cases.append(
            {
                "id": spec["id"],
                "gamma": spec["gamma"],
                "low_in": spec["low_in"],
                "high_in": spec["high_in"],
                "low_out": spec["low_out"],
                "high_out": spec["high_out"],
                "output_file": f"out/{spec['file']}",
                "stats": stats,
                "clipped_low_fraction": float(
                    round(np.mean(normalized < spec["low_in"]), 6)
                ),
                "clipped_high_fraction": float(
                    round(np.mean(normalized > spec["high_in"]), 6)
                ),
                "in_range_std_ratio": float(round(in_range_std_ratio, 4)),
            }
        )

    save_gamma_figure(gamma_results, OUT_DIR / FIGURE_FILES[3])
    save_transfer_curves(OUT_DIR / FIGURE_FILES[4])
    save_threshold_figure(
        image,
        processed_by_id["b"],
        image_stats_in,
        imadjust_cases[3]["stats"],
        "П. 3.2, б. imadjust: [0.2 0.6] -> [0 1], gamma = 1",
        OUT_DIR / FIGURE_FILES[5],
    )
    save_threshold_figure(
        image,
        processed_by_id["c"],
        image_stats_in,
        imadjust_cases[4]["stats"],
        "П. 3.2, в. imadjust: [0 1] -> [0.2 0.6], gamma = 1",
        OUT_DIR / FIGURE_FILES[6],
    )

    for case in imadjust_cases:
        stats = case["stats"]
        print(
            f"{case['id']}: gamma={case['gamma']:g}, "
            f"[{case['low_in']:g} {case['high_in']:g}] -> "
            f"[{case['low_out']:g} {case['high_out']:g}], "
            f"середнє={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"діапазон {stats['min']}..{stats['max']}"
        )

    print("\nП. 3.3. Локальна еквалізація (adapthisteq)")
    kernel_size = (
        max(image.shape[0] // CLAHE_TILES, 4),
        max(image.shape[1] // CLAHE_TILES, 4),
    )
    adaptive_float = equalize_adapthist(
        normalized,
        kernel_size=kernel_size,
        clip_limit=CLAHE_CLIP_LIMIT,
        nbins=CLAHE_NBINS,
    )
    adaptive = to_uint8(adaptive_float * 255.0)
    adaptive_stats = image_stats(adaptive)
    save_gray(OUT_DIR / "adapthisteq.png", adaptive)
    save_adaptive_comparison(
        image,
        equalized,
        adaptive,
        OUT_DIR / FIGURE_FILES[7],
    )
    print(
        f"Після adapthisteq: середнє {adaptive_stats['mean']:.2f}, "
        f"std {adaptive_stats['std']:.2f}, "
        f"ентропія {adaptive_stats['entropy']:.2f} біт"
    )

    print("\nП. 4. Моделі шуму")
    rng = np.random.default_rng(SEED)
    noise_cases = [
        {
            "id": "gaussian",
            "title": "Гаусів шум (mean = 0, variance = 0.01)",
            "short_title": "гаусів шум",
            "image": add_gaussian_noise(image, rng),
            "parameters": {
                "mean": GAUSSIAN_MEAN,
                "variance": GAUSSIAN_VARIANCE,
            },
        },
        {
            "id": "salt_pepper",
            "title": "Шум сіль і перець (density = 0.05)",
            "short_title": "сіль і перець",
            "image": add_salt_pepper_noise(image, rng),
            "parameters": {"density": SALT_PEPPER_DENSITY},
        },
        {
            "id": "poisson",
            "title": "Пуассонівський шум",
            "short_title": "пуассонівський шум",
            "image": add_poisson_noise(image, rng),
            "parameters": {"distribution": "poisson"},
        },
        {
            "id": "speckle",
            "title": "Спекл (мультиплікативний, variance = 0.04)",
            "short_title": "спекл",
            "image": add_speckle_noise(image, rng),
            "parameters": {"variance": SPECKLE_VARIANCE},
        },
    ]

    noise_records = []
    for case in noise_cases:
        stats = image_stats(case["image"])
        case["stats"] = stats
        save_gray(OUT_DIR / f"noise_{case['id']}.png", case["image"])
        changed_fraction = float(round(np.mean(case["image"] != image), 6))
        noise_records.append(
            {
                "id": case["id"],
                "parameters": case["parameters"],
                "output_file": f"out/noise_{case['id']}.png",
                "stats": stats,
                "changed_fraction": changed_fraction,
                "psnr_db": float(round(psnr(image, case["image"]), 4)),
            }
        )
        print(
            f"{case['id']}: середнє={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}, ентропія={stats['entropy']:.2f} біт, "
            f"змінено {100.0 * changed_fraction:.1f}% пікселів, "
            f"psnr={psnr(image, case['image']):.2f} дБ"
        )

    save_noise_figure(noise_cases, OUT_DIR / FIGURE_FILES[8])

    # збираємо результати в json
    results = {
        "title": "Покращення зображення за допомогою еквалізації",
        "seed": SEED,
        "input": {
            "file": "data/input.png",
            "stats": image_stats_in,
        },
        "histogram": {
            "bins": 256,
        },
        "histeq": {
            "levels": 256,
            "output_file": "out/histeq.png",
            "stats": equalized_stats,
        },
        "adapthisteq": {
            "clip_limit": CLAHE_CLIP_LIMIT,
            "nbins": CLAHE_NBINS,
            "tiles": CLAHE_TILES,
            "output_file": "out/adapthisteq.png",
            "stats": adaptive_stats,
        },
        "imadjust": {
            "cases": imadjust_cases,
        },
        "noise": {
            "cases": noise_records,
        },
        "figures": [f"out/{name}" for name in FIGURE_FILES],
    }

    results_path = OUT_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nГотово, результати в {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
