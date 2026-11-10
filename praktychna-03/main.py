# практична робота 3. покращення зображення методами просторової фільтрації
# запуск: python main.py

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from utils import (
    histogram_axis,
    image_axis,
    load_gray,
    mse,
    psnr,
    save_figure,
    ssim,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
INPUT_PATH = DATA_DIR / "input.png"

# параметри шуму і фільтрів
GAUSSIAN_VARIANCE = 0.01
GAUSSIAN_MEAN = 0.0
SALT_PEPPER_DENSITY = 0.05
MEDIAN_MASKS = (3, 7)
AVERAGE_MASKS = (3, 5, 7, 9)
RANK_MASK = 3
ADAPTIVE_SMAX = 7
ADAPTIVE_FALLBACK = "median"
BORDER_MODE = "reflect"
SEED_GAUSSIAN = 2026
SEED_IMPULSE = 2027


# шлях відносно папки практичної роботи
def rel_path(path):
    return Path(path).relative_to(BASE_DIR).as_posix()


# додаємо гаусівський шум
def add_gaussian_noise(image, variance=GAUSSIAN_VARIANCE, mean=GAUSSIAN_MEAN, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED_GAUSSIAN)
    sigma = math.sqrt(variance) * 255.0
    noise = rng.normal(loc=mean * 255.0, scale=sigma, size=image.shape)
    noisy = np.rint(image.astype(np.float64) + noise)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


# додаємо шум типу "сіль і перець"
def add_salt_pepper_noise(image, density=SALT_PEPPER_DENSITY, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED_IMPULSE)
    noisy = image.copy()
    field = rng.random(image.shape)
    half = density / 2.0
    noisy[field < half] = 0
    noisy[field > 1.0 - half] = 255
    return noisy


# медіанний фільтр
def median_filter(image, size=3):
    if size < 3 or size % 2 == 0:
        raise ValueError("маска має бути непарною і не меншою за 3")
    filtered = ndimage.median_filter(image, size=size, mode=BORDER_MODE)
    return filtered.astype(np.uint8)


# адаптивний медіанний фільтр
def adaptive_median_filter(image, smax=ADAPTIVE_SMAX, fallback_mode=ADAPTIVE_FALLBACK):
    if smax < 3 or smax % 2 == 0:
        raise ValueError("smax має бути непарним і не меншим за 3")

    radius_max = smax // 2
    # симетричне доповнення, щоб не перевіряти межі вікна
    padded = np.pad(image.astype(np.int32), radius_max, mode="symmetric")
    height, width = image.shape
    result = image.copy()

    for row in range(height):
        center_row = row + radius_max
        for col in range(width):
            center_col = col + radius_max
            zxy = int(padded[center_row, center_col])
            zmed = zxy
            window_size = 3

            while window_size <= smax:
                radius = window_size // 2
                window = padded[
                    center_row - radius : center_row + radius + 1,
                    center_col - radius : center_col + radius + 1,
                ]
                # з відсортованого вікна беремо мінімум, максимум і медіану
                values = np.sort(window, axis=None)
                zmin = int(values[0])
                zmax = int(values[-1])
                zmed = int(values[values.size // 2])

                # медіана не схожа на імпульс, перевіряємо центр
                if zmin < zmed < zmax:
                    if zmin < zxy < zmax:
                        result[row, col] = zxy
                    else:
                        result[row, col] = zmed
                    break

                window_size += 2
            else:
                # вікно вже більше за smax
                result[row, col] = zxy if fallback_mode == "center" else zmed

    return result.astype(np.uint8)


# ранговий фільтр
def rank_filter(image, size=RANK_MASK, order=1):
    total = size * size
    if not 1 <= order <= total:
        raise ValueError("order має бути від 1 до %d" % total)
    # scipy нумерує ранги з нуля
    filtered = ndimage.rank_filter(image, rank=order - 1, size=size, mode=BORDER_MODE)
    return filtered.astype(np.uint8)


# фільтр мінімуму
def minimum_filter(image, size=RANK_MASK):
    return rank_filter(image, size=size, order=1)


# фільтр максимуму
def maximum_filter(image, size=RANK_MASK):
    return rank_filter(image, size=size, order=size * size)


# фільтр усереднення
def average_filter(image, size=3):
    kernel = np.full((size, size), 1.0 / (size * size), dtype=np.float64)
    filtered = ndimage.correlate(image.astype(np.float64), kernel, mode=BORDER_MODE)
    return np.clip(np.rint(filtered), 0.0, 255.0).astype(np.uint8)


# метрики відносно початкового зображення
def compute_metrics(reference, image):
    return {
        "mse": round(float(mse(reference, image)), 6),
        "psnr": round(float(psnr(reference, image)), 6),
        "ssim": round(float(ssim(reference, image)), 8),
    }


# запис одного результату для json
def make_record(identifier, label, filter_name, params, image, reference, figure_name):
    record = {"id": identifier, "label": label, "filter": filter_name}
    record.update(params)
    record["figure"] = figure_name
    record.update(compute_metrics(reference, image))
    return record


# рисунок зображення і гістограми
def save_single_figure(image, path, title, histogram_title, metrics=None):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.6, 4.4),
        gridspec_kw={"width_ratios": (1.0, 1.15)},
    )
    image_axis(axes[0], image, title)
    histogram_axis(axes[1], image, histogram_title)
    if metrics is None:
        subtitle = (
            f"Розмір: {image.shape[1]}x{image.shape[0]}   |   "
            f"Яскравість: від {int(image.min())} до {int(image.max())}"
        )
    else:
        subtitle = f"PSNR = {metrics['psnr']:.2f} дБ   |   SSIM = {metrics['ssim']:.4f}"
    figure.suptitle(subtitle, fontsize=10)
    save_figure(figure, path)


# сітка зображень для порівняння
def save_comparison(entries, path, columns, suptitle, summary_text=None):
    total = len(entries) + (1 if summary_text else 0)
    rows = math.ceil(total / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.2 * rows))
    flat_axes = np.atleast_2d(axes).ravel()

    for axis, (image, title) in zip(flat_axes, entries):
        image_axis(axis, image, title)

    if summary_text:
        summary_axis = flat_axes[len(entries)]
        summary_axis.axis("off")
        summary_axis.text(
            0.0,
            1.0,
            summary_text,
            va="top",
            ha="left",
            fontsize=9.5,
            linespacing=1.6,
        )

    for axis in flat_axes[total:]:
        axis.axis("off")

    figure.suptitle(suptitle, fontsize=13)
    save_figure(figure, path)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original = load_gray(INPUT_PATH)

    rng_gaussian = np.random.default_rng(SEED_GAUSSIAN)
    rng_impulse = np.random.default_rng(SEED_IMPULSE)
    gaussian_noisy = add_gaussian_noise(original, rng=rng_gaussian)
    impulse_noisy = add_salt_pepper_noise(original, rng=rng_impulse)

    results = []
    figures = []

    def add_result(identifier, label, filter_name, params, image, figure_name, title, hist_title):
        # рахуємо метрики, зберігаємо рисунок і пишемо запис у json
        record = make_record(
            identifier,
            label,
            filter_name,
            params,
            image,
            original,
            rel_path(OUT_DIR / figure_name),
        )
        results.append(record)
        save_single_figure(image, OUT_DIR / figure_name, title, hist_title, record)
        figures.append(rel_path(OUT_DIR / figure_name))

    # початкове зображення
    save_single_figure(
        original,
        OUT_DIR / "01_original.png",
        "Початкове зображення",
        "Гістограма початкового зображення",
    )
    figures.append(rel_path(OUT_DIR / "01_original.png"))

    # гаусівський шум
    add_result(
        "gaussian_noise",
        "Гаусівський шум",
        "none",
        {"noise": "gaussian", "variance": GAUSSIAN_VARIANCE},
        gaussian_noisy,
        "02_gaussian_noisy.png",
        "Зображення з гаусівським шумом (sigma = 25,5)",
        "Гістограма зашумленого зображення",
    )

    # медіанний фільтр
    median_images = {}
    for size, figure_name in ((3, "04_median_3x3.png"), (7, "05_median_7x7.png")):
        filtered = median_filter(gaussian_noisy, size=size)
        median_images[size] = filtered
        add_result(
            f"median_{size}x{size}",
            f"Медіанний фільтр {size}x{size}",
            "median",
            {"noise": "gaussian", "mask": f"{size}x{size}"},
            filtered,
            figure_name,
            f"Медіанний фільтр, маска {size}x{size}",
            f"Гістограма після медіанного фільтра {size}x{size}",
        )

    # адаптивний медіанний фільтр
    adaptive_image = adaptive_median_filter(gaussian_noisy)
    add_result(
        "adaptive_median_smax7",
        "Адаптивна медіана, Smax = 7",
        "adaptive_median",
        {"noise": "gaussian", "smax": ADAPTIVE_SMAX},
        adaptive_image,
        "06_adaptive_median_smax7.png",
        f"Адаптивний медіанний фільтр, Smax = {ADAPTIVE_SMAX}",
        "Гістограма після адаптивного медіанного фільтра",
    )

    # фільтр мінімуму
    min_image = minimum_filter(gaussian_noisy, size=RANK_MASK)
    add_result(
        "min_3x3",
        "Фільтр мінімуму 3x3",
        "rank_min",
        {"noise": "gaussian", "mask": f"{RANK_MASK}x{RANK_MASK}", "order": 1},
        min_image,
        "07_min_filter_3x3.png",
        "Фільтр мінімуму, маска 3x3 (order = 1)",
        "Гістограма після фільтра мінімуму",
    )

    # фільтр максимуму
    max_image = maximum_filter(gaussian_noisy, size=RANK_MASK)
    add_result(
        "max_3x3",
        "Фільтр максимуму 3x3",
        "rank_max",
        {
            "noise": "gaussian",
            "mask": f"{RANK_MASK}x{RANK_MASK}",
            "order": RANK_MASK * RANK_MASK,
        },
        max_image,
        "08_max_filter_3x3.png",
        "Фільтр максимуму, маска 3x3 (order = 9)",
        "Гістограма після фільтра максимуму",
    )

    # фільтр усереднення
    average_images = {}
    for index, size in enumerate(AVERAGE_MASKS):
        filtered = average_filter(gaussian_noisy, size=size)
        average_images[size] = filtered
        add_result(
            f"average_{size}x{size}",
            f"Усереднення {size}x{size}",
            "average",
            {"noise": "gaussian", "mask": f"{size}x{size}"},
            filtered,
            f"{9 + index:02d}_average_{size}x{size}.png",
            f"Фільтр усереднення, маска {size}x{size}",
            f"Гістограма після усереднення {size}x{size}",
        )

    # додатковий експеримент з шумом "сіль і перець"
    add_result(
        "salt_pepper_noise",
        "Шум «сіль і перець» 5 %",
        "none",
        {"noise": "salt_pepper", "density": SALT_PEPPER_DENSITY},
        impulse_noisy,
        "03_salt_pepper_noisy.png",
        "Зображення з шумом «сіль і перець» (5 %)",
        "Гістограма з імпульсним шумом",
    )

    # для цього шуму окремі рисунки не робимо, все є у порівнянні
    impulse_images = {
        "median_3x3": median_filter(impulse_noisy, size=3),
        "median_7x7": median_filter(impulse_noisy, size=7),
        "adaptive_median_smax7": adaptive_median_filter(impulse_noisy),
        "average_3x3": average_filter(impulse_noisy, size=3),
    }
    impulse_params = {
        "median_3x3": {"mask": "3x3"},
        "median_7x7": {"mask": "7x7"},
        "adaptive_median_smax7": {"smax": ADAPTIVE_SMAX},
        "average_3x3": {"mask": "3x3"},
    }
    impulse_labels = {
        "median_3x3": "Медіанний фільтр 3x3",
        "median_7x7": "Медіанний фільтр 7x7",
        "adaptive_median_smax7": "Адаптивна медіана, Smax = 7",
        "average_3x3": "Усереднення 3x3",
    }
    impulse_filters = {
        "median_3x3": "median",
        "median_7x7": "median",
        "adaptive_median_smax7": "adaptive_median",
        "average_3x3": "average",
    }
    impulse_figure = rel_path(OUT_DIR / "14_impulse_comparison.png")
    for identifier, filtered in impulse_images.items():
        params = {"noise": "salt_pepper"}
        params.update(impulse_params[identifier])
        results.append(
            make_record(
                identifier,
                impulse_labels[identifier],
                impulse_filters[identifier],
                params,
                filtered,
                original,
                impulse_figure,
            )
        )

    # порівняння на гаусівському шумі
    gaussian_by_id = {
        record["id"]: record
        for record in results
        if record.get("noise") == "gaussian"
    }
    gaussian_entries = [
        (original, "Оригінал"),
        (
            gaussian_noisy,
            f"Гаусівський шум\nPSNR {gaussian_by_id['gaussian_noise']['psnr']:.2f} дБ",
        ),
        (
            median_images[3],
            f"Медіана 3x3\nPSNR {gaussian_by_id['median_3x3']['psnr']:.2f} дБ",
        ),
        (
            median_images[7],
            f"Медіана 7x7\nPSNR {gaussian_by_id['median_7x7']['psnr']:.2f} дБ",
        ),
        (
            adaptive_image,
            f"Адаптивна медіана\nPSNR {gaussian_by_id['adaptive_median_smax7']['psnr']:.2f} дБ",
        ),
        (
            min_image,
            f"Мінімум 3x3\nPSNR {gaussian_by_id['min_3x3']['psnr']:.2f} дБ",
        ),
        (
            max_image,
            f"Максимум 3x3\nPSNR {gaussian_by_id['max_3x3']['psnr']:.2f} дБ",
        ),
        (
            average_images[3],
            f"Усереднення 3x3\nPSNR {gaussian_by_id['average_3x3']['psnr']:.2f} дБ",
        ),
        (
            average_images[5],
            f"Усереднення 5x5\nPSNR {gaussian_by_id['average_5x5']['psnr']:.2f} дБ",
        ),
        (
            average_images[7],
            f"Усереднення 7x7\nPSNR {gaussian_by_id['average_7x7']['psnr']:.2f} дБ",
        ),
        (
            average_images[9],
            f"Усереднення 9x9\nPSNR {gaussian_by_id['average_9x9']['psnr']:.2f} дБ",
        ),
    ]
    best_average = max(
        (gaussian_by_id[f"average_{size}x{size}"] for size in AVERAGE_MASKS),
        key=lambda record: record["psnr"],
    )
    summary = (
        "Підсумок (гаусівський шум)\n\n"
        f"Найкращий PSNR:\n{best_average['label']}: "
        f"{best_average['psnr']:.2f} дБ\n\n"
        f"Медіана 3x3: {gaussian_by_id['median_3x3']['psnr']:.2f} дБ,\n"
        f"SSIM {gaussian_by_id['median_3x3']['ssim']:.4f}\n\n"
        f"Адаптивна медіана: "
        f"{gaussian_by_id['adaptive_median_smax7']['psnr']:.2f} дБ,\n"
        f"SSIM {gaussian_by_id['adaptive_median_smax7']['ssim']:.4f}\n\n"
        "Для гаусівського шуму переважає лінійне усереднення."
    )
    save_comparison(
        gaussian_entries,
        OUT_DIR / "13_gaussian_comparison.png",
        columns=4,
        suptitle="Порівняння фільтрів на зображенні з гаусівським шумом",
        summary_text=summary,
    )
    figures.append(rel_path(OUT_DIR / "13_gaussian_comparison.png"))

    # порівняння на шумі "сіль і перець"
    impulse_by_id = {
        record["id"]: record
        for record in results
        if record.get("noise") == "salt_pepper"
    }
    impulse_entries = [
        (original, "Оригінал"),
        (
            impulse_noisy,
            f"Шум «сіль і перець»\nPSNR {impulse_by_id['salt_pepper_noise']['psnr']:.2f} дБ",
        ),
        (
            impulse_images["median_3x3"],
            f"Медіана 3x3\nPSNR {impulse_by_id['median_3x3']['psnr']:.2f} дБ",
        ),
        (
            impulse_images["median_7x7"],
            f"Медіана 7x7\nPSNR {impulse_by_id['median_7x7']['psnr']:.2f} дБ",
        ),
        (
            impulse_images["adaptive_median_smax7"],
            f"Адаптивна медіана\n"
            f"PSNR {impulse_by_id['adaptive_median_smax7']['psnr']:.2f} дБ",
        ),
        (
            impulse_images["average_3x3"],
            f"Усереднення 3x3\nPSNR {impulse_by_id['average_3x3']['psnr']:.2f} дБ",
        ),
    ]
    save_comparison(
        impulse_entries,
        OUT_DIR / "14_impulse_comparison.png",
        columns=3,
        suptitle="Додатковий експеримент: шум «сіль і перець» (5 %)",
    )
    figures.append(rel_path(OUT_DIR / "14_impulse_comparison.png"))
    figures.sort()

    # зберігаємо результати
    payload = {
        "work": "praktychna-03",
        "title": "Покращення зображення методами просторової фільтрації",
        "input": {
            "path": rel_path(INPUT_PATH),
            "width": int(original.shape[1]),
            "height": int(original.shape[0]),
        },
        "parameters": {
            "seed_gaussian": SEED_GAUSSIAN,
            "seed_impulse": SEED_IMPULSE,
            "gaussian_noise": {
                "mean": GAUSSIAN_MEAN,
                "variance": GAUSSIAN_VARIANCE,
            },
            "salt_pepper_density": SALT_PEPPER_DENSITY,
            "median_masks": [f"{size}x{size}" for size in MEDIAN_MASKS],
            "average_masks": [f"{size}x{size}" for size in AVERAGE_MASKS],
            "adaptive_smax": ADAPTIVE_SMAX,
            "rank_mask": f"{RANK_MASK}x{RANK_MASK}",
            "border_mode": BORDER_MODE,
        },
        "results": results,
        "figures": figures,
    }
    results_path = OUT_DIR / "results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    # щоб український текст не ламався у консолі
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    run()
    print("результати збережено в out/results.json")


if __name__ == "__main__":
    main()
