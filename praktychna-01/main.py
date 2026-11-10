# практична робота 1: робота з графічними файлами
# запуск: python main.py

import json
from pathlib import Path

import numpy as np

from utils import (
    image_axis,
    load_gray,
    mse,
    new_figure,
    psnr,
    save_figure,
    save_jpeg,
    setup_console,
    ssim,
)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "out"
IMAGE_NAME = "input.png"
JPEG_QUALITIES = (25, 5, 0)


def format_number(value):
    # числа з пробілом між розрядами
    return f"{int(value):,}".replace(",", " ")


def rel_path(path):
    # шлях відносно папки проєкту
    return path.relative_to(BASE).as_posix()


def print_whos(image):
    # таблиця whos
    size = "x".join(str(int(value)) for value in image.shape)
    print(f"{'Name':<10}{'Size':<16}{'Bytes':>10}  Class")
    print(f"{'i':<10}{size:<16}{image.nbytes:>10}  {image.dtype.name}")
    print(f"Grand total is {image.size} elements using {image.nbytes} bytes")


def display_arrays(image):
    # два варіанти відображення: 0-255 і min-max
    low = int(image.min())
    high = int(image.max())
    if high <= low:
        raise ValueError("діапазон яскравостей вироджений")
    full = image.astype(np.float64) / 255.0
    stretched = (image.astype(np.float64) - low) / float(high - low)
    return full, stretched, low, high


def make_figure_5(image, m, n, low, high, path):
    # рисунок для пункту 5: imshow(i)
    figure, axis = new_figure(figsize=(5.6, 6.4))
    image_axis(
        axis,
        image,
        f"П.5. imshow(i): діапазон 0-255\nM={m}, N={n}, min={low}, max={high}",
        vmin=0,
        vmax=255,
    )
    save_figure(figure, path)


def make_figure_6(image, low, high, path):
    # порівняння imshow(i) та imshow(i, [])
    figure, axes = new_figure(1, 2, figsize=(11.0, 6.6))
    image_axis(
        axes[0],
        image,
        "imshow(i): повний діапазон 0-255",
        vmin=0,
        vmax=255,
    )
    image_axis(
        axes[1],
        image,
        f"imshow(i, []): діапазон {low}-{high}",
        vmin=low,
        vmax=high,
    )
    figure.suptitle(
        "П.6. Порівняння imshow(i) та imshow(i, []): розтягування діапазону",
        fontsize=12,
    )
    save_figure(figure, path)


def make_figure_8(image, low, high, records, decoded, path):
    # оригінал і три jpeg у сітці 2x2
    figure, axes = new_figure(2, 2, figsize=(11.0, 10.0))
    image_axis(
        axes[0, 0],
        image,
        f"Оригінал PNG\n{format_number(image.nbytes)} байт",
        vmin=low,
        vmax=high,
    )
    for axis, record in zip(axes.ravel()[1:], records):
        quality = record["quality"]
        title = (
            f"JPEG, q={quality}\n"
            f"{format_number(record['file_size_bytes'])} байт, "
            f"PSNR {record['psnr_db']:.2f} дБ"
        )
        image_axis(axis, decoded[quality], title, vmin=low, vmax=high)
    figure.suptitle(
        f"П.8. Порівняння JPEG різної якості (діапазон відображення {low}-{high})",
        fontsize=12,
    )
    save_figure(figure, path)


def save_jpeg_variants(image):
    # зберігаємо jpeg з різною якістю і рахуємо метрики
    records = []
    decoded = {}
    for quality in JPEG_QUALITIES:
        jpeg_path = OUT_DIR / f"input_q{quality}.jpg"
        save_jpeg(jpeg_path, image, quality)
        restored = load_gray(jpeg_path)
        decoded[quality] = restored
        records.append(
            {
                "quality": quality,
                "path": rel_path(jpeg_path),
                "file_size_bytes": jpeg_path.stat().st_size,
                "psnr_db": round(psnr(image, restored), 6),
                "mse": round(mse(image, restored), 6),
                "ssim": round(ssim(image, restored), 6),
            }
        )
    return records, decoded


def main():
    setup_console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Практична робота 1. Робота з графічними файлами")
    print("=" * 60)

    # пункт 1: читаємо картинку
    print("\nПункт 1. Зчитання зображення")
    data_path = DATA_DIR / IMAGE_NAME
    image = load_gray(data_path)
    print(f"  Файл: {rel_path(data_path)}")
    print(f"  Клас: {image.dtype.name}, форма: {image.shape}")
    print(f"  Кількість елементів: {format_number(image.size)}")

    # пункт 2: рахуємо розмір
    print("\nПункт 2. Розмір зображення (аналог size)")
    m, n = image.shape
    print(f"  M = {m} (кількість рядків)")
    print(f"  N = {n} (кількість стовпчиків)")
    print(f"  size(i) = {m} x {n}")

    # пункт 3: рахуємо об'єм
    print("\nПункт 3. Об'єм зображення")
    channels = 1 if image.ndim == 2 else image.shape[2]
    itemsize = image.dtype.itemsize
    volume_bytes = m * n * channels * itemsize
    print(
        f"  V = M * N * channels * itemsize = {m} * {n} * {channels} * {itemsize}"
        f" = {format_number(volume_bytes)} байт"
    )

    # пункт 4: показуємо whos
    print("\nПункт 4. Таблиця whos")
    print_whos(image)

    # пункт 5: малюємо зображення
    print("\nПункт 5. Виведення зображення (аналог imshow)")
    low = int(image.min())
    high = int(image.max())
    figure5 = OUT_DIR / "fig05_imshow.png"
    make_figure_5(image, m, n, low, high, figure5)
    print(f"  Рисунок збережено: {rel_path(figure5)}")

    # пункт 6: порівнюємо два режими відображення
    print("\nПункт 6. Порівняння imshow(i) та imshow(i, [])")
    full_range, minmax_range, low, high = display_arrays(image)
    difference = np.abs(full_range - minmax_range)
    mean_difference = float(difference.mean())
    max_difference = float(difference.max())
    print("  imshow(i): діапазон відображення 0-255")
    print(f"  imshow(i, []): діапазон відображення {low}-{high}")
    print(f"  Середня абсолютна різниця нормованих масивів: {mean_difference:.6f}")
    print(f"  Максимальна абсолютна різниця: {max_difference:.6f}")
    figure6 = OUT_DIR / "fig06_imshow_vs_imshow_empty.png"
    make_figure_6(image, low, high, figure6)
    print(f"  Рисунок збережено: {rel_path(figure6)}")

    # пункт 7: зберігаємо jpeg
    print("\nПункт 7. Збереження JPEG з якістю q = 25, 5, 0")
    jpeg_records, decoded = save_jpeg_variants(image)
    for record in jpeg_records:
        print(
            f"  q={record['quality']}: {record['path']}, "
            f"{format_number(record['file_size_bytes'])} байт, "
            f"PSNR {record['psnr_db']:.2f} дБ"
        )

    # пункт 8: порівнюємо оригінал і jpeg
    print("\nПункт 8. Порівняння збережених JPEG-зображень")
    figure8 = OUT_DIR / "fig08_jpeg_comparison.png"
    make_figure_8(image, low, high, jpeg_records, decoded, figure8)
    print(f"  Рисунок збережено: {rel_path(figure8)}")

    # пишемо результати у json
    results = {
        "image_path": rel_path(data_path),
        "M": m,
        "N": n,
        "channels": channels,
        "dtype": image.dtype.name,
        "itemsize": itemsize,
        "volume_bytes": volume_bytes,
        "elements": int(image.size),
        "min": low,
        "max": high,
        "mean": round(float(image.mean()), 4),
        "std": round(float(image.std()), 4),
        "display_ranges": {
            "imshow_default": [0, 255],
            "imshow_empty": [low, high],
        },
        "display_difference": {
            "mean": round(mean_difference, 6),
            "max": round(max_difference, 6),
        },
        "figures": {
            "item5": rel_path(figure5),
            "item6": rel_path(figure6),
            "item8": rel_path(figure8),
        },
        "jpeg": jpeg_records,
    }

    results_path = OUT_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nРезультати збережено: {rel_path(results_path)}")


if __name__ == "__main__":
    main()
