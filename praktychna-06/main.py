# практична робота 6: вимір параметрів графічних об'єктів
# запуск: python main.py

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from skimage.measure import regionprops as skimage_regionprops
from skimage.morphology import local_maxima, skeletonize, thin

from utils import image_axis, load_gray, save_figure, save_gray

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
INPUT_PATH = DATA_DIR / "binary_shapes.png"

# поріг для bwareaopen
AREA_OPEN_THRESHOLD = 40

# маска 8-зв'язності (квадрат 3x3) і 4-зв'язності (хрест)
STRUCT8 = np.ones((3, 3), dtype=bool)
STRUCT4 = ndi.generate_binary_structure(2, 1)

# ваги bwarea для кожного шаблону 2x2 (код 0..15)
BAREA_WEIGHTS = np.array(
    [
        0.0,   # 0000
        0.25,  # 0001
        0.25,  # 0010
        0.5,   # 0011
        0.25,  # 0100
        0.5,   # 0101
        0.75,  # 0110
        0.75,  # 0111
        0.25,  # 1000
        0.75,  # 1001
        0.5,   # 1010
        0.75,  # 1011
        0.5,   # 1100
        0.75,  # 1101
        0.75,  # 1110
        1.0,   # 1111
    ],
    dtype=np.float64,
)

FIGURE_FILES = [
    "01_input_binary.png",
    "02_bwarea.png",
    "03_bwareaopen.png",
    "04_bwlabel.png",
    "05_bwlabeln_3d.png",
    "06_bwdist.png",
    "07_bweuler.png",
    "08_bwfill.png",
    "09_bwmorph_basic.png",
    "10_bwmorph_skeleton.png",
    "11_bwmorph_test_pattern.png",
    "12_bwperim.png",
    "13_bwselect.png",
    "14_bwulterode.png",
    "15_regionprops_overlay.png",
    "16_regionprops_table.png",
]

MASK_FILES = [
    "bwareaopen.png",
    "bwfill.png",
    "bwperim8.png",
    "bwperim4.png",
    "bwselect.png",
    "bwulterode.png",
    "skel.png",
    "thin.png",
    "bwmorph_test.png",
]

MORPH_MAIN_OPERATIONS = [
    "erode",
    "dilate",
    "open",
    "close",
    "remove",
    "bridge",
    "clean",
    "fill",
    "spur",
    "majority",
]

MORPH_SUPPORTED = MORPH_MAIN_OPERATIONS + ["skel", "thin"]

MORPH_UNSUPPORTED = [
    "shrink",
    "thicken",
    "diag",
    "hbreak",
    "tophat",
    "bothat",
    "endpoints",
]

CSV_FIELDS = [
    "label",
    "area",
    "centroid_row",
    "centroid_col",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "perimeter",
    "eccentricity",
    "orientation_deg",
    "major_axis_length",
    "minor_axis_length",
    "equivalent_diameter",
    "euler_number",
    "filled_area",
]


# допоміжні перетворення
def binary_to_gray(mask):
    # маска -> зображення 0/255
    return np.asarray(mask, dtype=bool).astype(np.uint8) * 255


def connectivity_structure(ndim, connectivity):
    # структура зв'язності: 4/6 -> 1 крок, 8/18 -> 2, 26 -> 3
    hops = {4: 1, 8: 2, 6: 1, 18: 2, 26: 3}.get(int(connectivity))
    if hops is None:
        raise ValueError("непідтримувана зв'язність: %s" % connectivity)
    if hops > ndim:
        raise ValueError("зв'язність %s не має сенсу для %d-D" % (connectivity, ndim))
    return ndi.generate_binary_structure(ndim, hops)


def highlight_mask(base, highlight, color=(220, 0, 0), foreground=80, background=255):
    # сірий об'єкт, кольорові виділені пікселі
    base = np.asarray(base, dtype=bool)
    highlight = np.asarray(highlight, dtype=bool) & base
    gray = np.where(base, foreground, background).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray]).copy()
    rgb[highlight] = color
    return rgb


def overlay_changes(before, after):
    # зелене - додано, червоне - прибрано
    before = np.asarray(before, dtype=bool)
    after = np.asarray(after, dtype=bool)
    gray = np.where(before, 80, 255).astype(np.uint8)
    rgb = np.dstack([gray, gray, gray]).copy()
    rgb[after & ~before] = (0, 150, 0)
    rgb[before & ~after] = (210, 0, 0)
    return rgb


def snap_seed_to_foreground(mask, row, col):
    # центр мас може лежати в отворі, прив'язуємо до найближчого пікселя
    height, width = mask.shape
    row_int = int(np.clip(round(row), 0, height - 1))
    col_int = int(np.clip(round(col), 0, width - 1))
    if mask[row_int, col_int]:
        return row_int, col_int
    rows, cols = np.nonzero(mask)
    distances = (rows - row) ** 2 + (cols - col) ** 2
    index = int(np.argmin(distances))
    return int(rows[index]), int(cols[index])


# аналоги функцій matlab
def bwarea_weighted(mask):
    # зважена площа за вікнами 2x2
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    # рамка з нулів: поза зображенням вважається фон
    padded = np.zeros((height + 2, width + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    code = (
        padded[:-1, :-1]
        + 2 * padded[:-1, 1:]
        + 4 * padded[1:, :-1]
        + 8 * padded[1:, 1:]
    )
    return float(BAREA_WEIGHTS[code].sum())


def relabel_by_first_occurrence(labels, count):
    # нумерація в порядку першого пікселя за стовпцями, як у matlab
    flat = labels.ravel(order="F")
    values, first_index = np.unique(flat, return_index=True)
    nonzero = values > 0
    values = values[nonzero]
    first_index = first_index[nonzero]
    order = np.argsort(first_index, kind="stable")
    sorted_values = values[order]
    mapping = np.zeros(count + 1, dtype=np.int32)
    mapping[sorted_values] = np.arange(1, sorted_values.size + 1, dtype=np.int32)
    return mapping[labels], int(sorted_values.size)


def bwlabel_matlab(mask, connectivity=8):
    # мітки 2-d компонент
    mask = np.asarray(mask, dtype=bool)
    structure = connectivity_structure(2, connectivity)
    labels, count = ndi.label(mask, structure=structure)
    return relabel_by_first_occurrence(labels, int(count))


def bwlabeln_matlab(mask, connectivity=26):
    # мітки компонент у багатовимірному масиві
    mask = np.asarray(mask, dtype=bool)
    structure = connectivity_structure(mask.ndim, connectivity)
    labels, count = ndi.label(mask, structure=structure)
    return relabel_by_first_occurrence(labels, int(count))


def bwareaopen_matlab(mask, min_area, connectivity=8):
    # видаляємо компоненти з площею меншою за min_area
    labels, count = bwlabel_matlab(mask, connectivity)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    kept = [i for i in range(1, count + 1) if areas[i] >= min_area]
    removed = [i for i in range(1, count + 1) if areas[i] < min_area]
    result = np.isin(labels, kept)
    return result, labels, kept, removed, areas[1:]


def count_holes(mask, object_connectivity=8):
    # отвір - компонент фону, який не виходить на рамку зображення
    mask = np.asarray(mask, dtype=bool)
    background_structure = STRUCT4 if object_connectivity == 8 else STRUCT8
    padded = np.pad(~mask, 1, mode="constant", constant_values=True)
    labels, _ = ndi.label(padded, structure=background_structure)
    outside = int(labels[0, 0])
    inner = labels[1:-1, 1:-1]
    hole_mask = (~mask) & (inner > 0) & (inner != outside)
    hole_labels, hole_count = ndi.label(hole_mask, structure=STRUCT8)
    areas = np.bincount(hole_labels.ravel(), minlength=hole_count + 1)[1:]
    return hole_mask, int(hole_count), sorted((int(a) for a in areas), reverse=True)


def bwfill_holes(mask, connectivity=8):
    # заповнюємо отвори
    hole_mask, _, areas = count_holes(mask, connectivity)
    return np.asarray(mask, dtype=bool) | hole_mask, hole_mask, areas


def bweuler_matlab(mask):
    # ейлер = об'єкти - отвори
    _, components = bwlabel_matlab(mask, 8)
    _, holes, _ = count_holes(mask, 8)
    return int(components - holes), int(components), int(holes)


def bwdist_matlab(mask):
    # евклідова відстань і мітки найближчого пікселя
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("bwdist потребує хоча б одного пікселя об'єкта")
    distance, indices = ndi.distance_transform_edt(
        ~mask, return_indices=True
    )
    nearest_rows = indices[0].astype(np.int64)
    nearest_cols = indices[1].astype(np.int64)
    linear = nearest_rows + nearest_cols * mask.shape[0] + 1
    return distance, linear, nearest_rows, nearest_cols


def bwmorph_bridge(mask):
    # з'єднуємо різні компоненти через фон завширшки 1
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    labels, _ = ndi.label(mask, structure=STRUCT8)
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    background = ~mask
    added = np.zeros_like(mask)
    pairs = [
        (padded[:-2, 1:-1], padded[2:, 1:-1]),
        (padded[1:-1, :-2], padded[1:-1, 2:]),
        (padded[:-2, :-2], padded[2:, 2:]),
        (padded[:-2, 2:], padded[2:, :-2]),
    ]
    for first, second in pairs:
        added |= background & (first > 0) & (second > 0) & (first != second)
    return mask | added


def bwmorph_matlab(mask, operation, iterations=1):
    # підмножина операцій matlab
    op = operation.lower()
    if op not in MORPH_SUPPORTED:
        raise ValueError("операція bwmorph '%s' не підтримується" % operation)
    source = np.asarray(mask, dtype=bool)
    if op == "skel":
        return skeletonize(source)
    if op == "thin":
        return thin(source)

    result = source.copy()
    repeat = max(1, int(iterations))
    for _ in range(repeat):
        if op == "erode":
            result = ndi.binary_erosion(
                result, structure=STRUCT8, border_value=0
            )
        elif op == "dilate":
            result = ndi.binary_dilation(
                result, structure=STRUCT8, border_value=0
            )
        elif op == "open":
            eroded = ndi.binary_erosion(
                result, structure=STRUCT8, border_value=0
            )
            result = ndi.binary_dilation(
                eroded, structure=STRUCT8, border_value=0
            )
        elif op == "close":
            dilated = ndi.binary_dilation(
                result, structure=STRUCT8, border_value=0
            )
            result = ndi.binary_erosion(
                dilated, structure=STRUCT8, border_value=0
            )
        elif op == "remove":
            interior = ndi.binary_erosion(result, structure=STRUCT4)
            result = result & ~interior
        elif op == "bridge":
            result = bwmorph_bridge(result)
        elif op == "clean":
            neighbors = ndi.convolve(
                result.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                mode="constant",
                cval=0,
            )
            result = result & (neighbors >= 2)
        elif op == "fill":
            neighbors = ndi.convolve(
                result.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                mode="constant",
                cval=0,
            )
            result = result | ((~result) & (neighbors == 8))
        elif op == "spur":
            neighbors = ndi.convolve(
                result.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                mode="constant",
                cval=0,
            )
            result = result & ~(neighbors == 2)
        elif op == "majority":
            neighbors = ndi.convolve(
                result.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                mode="constant",
                cval=0,
            )
            result = neighbors >= 5
    return result


def bwperim_matlab(mask, connectivity=8):
    # граничні пікселі об'єктів
    mask = np.asarray(mask, dtype=bool)
    structure = STRUCT8 if connectivity == 8 else STRUCT4
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def bwselect_matlab(labels, seeds):
    # вибираємо компоненти за затравочними точками (рядок, стовпець)
    height, width = labels.shape
    selected_labels = []
    for row, col in seeds:
        if 0 <= row < height and 0 <= col < width:
            value = int(labels[row, col])
            if value > 0 and value not in selected_labels:
                selected_labels.append(value)
    return np.isin(labels, selected_labels), selected_labels


def bwulterode_iterative(mask):
    # гранична ерозія послідовною ерозією квадратом 3x3
    mask = np.asarray(mask, dtype=bool)
    labels, count = bwlabel_matlab(mask, 8)
    ultimate = np.zeros_like(mask)
    records = []
    max_iterations = 0

    for label_value in range(1, count + 1):
        current = labels == label_value
        iterations = 0
        while current.any():
            eroded = ndi.binary_erosion(
                current, structure=STRUCT8, border_value=0
            )
            iterations += 1
            if not eroded.any():
                ultimate |= current
                break
            pieces, piece_count = ndi.label(current, structure=STRUCT8)
            for piece_index in range(1, piece_count + 1):
                piece = pieces == piece_index
                if not np.any(eroded & piece):
                    ultimate |= piece
            current = eroded
        points = int((ultimate & (labels == label_value)).sum())
        records.append(
            {
                "label": int(label_value),
                "iterations": int(iterations),
                "points": points,
            }
        )
        max_iterations = max(max_iterations, iterations)
    return ultimate, records, int(max_iterations)


def bwulterode_edt(mask):
    # довідковий варіант: локальні максимуми edt
    mask = np.asarray(mask, dtype=bool)
    distance = ndi.distance_transform_edt(mask)
    maxima = local_maxima(distance) & mask
    return maxima, distance


def build_regionprops(labels, count):
    # таблиця морфометричних ознак
    properties = skimage_regionprops(labels)
    rows = []
    for item in properties:
        label_value = int(item.label)
        component = labels == label_value
        rows_pixels, cols_pixels = np.nonzero(component)
        x = cols_pixels.astype(np.float64)
        y = rows_pixels.astype(np.float64)
        x_center = float(x.mean())
        y_center = float(y.mean())
        dx = x - x_center
        dy = y - y_center
        mu20 = float(np.mean(dx * dx))
        mu02 = float(np.mean(dy * dy))
        mu11 = float(np.mean(dx * dy))

        if rows_pixels.size > 1 and (mu20 > 0.0 or mu02 > 0.0):
            covariance = np.cov(np.vstack([x, y]), bias=True)
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            largest = max(float(eigenvalues[0]), 0.0)
            smallest = max(float(eigenvalues[1]), 0.0)
            eccentricity = (
                float(np.sqrt(1.0 - smallest / largest))
                if largest > 0.0
                else 0.0
            )
            major_length = 4.0 * float(np.sqrt(largest))
            minor_length = 4.0 * float(np.sqrt(smallest))
        else:
            eccentricity = 0.0
            major_length = 0.0
            minor_length = 0.0

        orientation = float(
            np.degrees(0.5 * np.arctan2(2.0 * mu11, mu20 - mu02))
        )

        bbox = item.bbox
        row_min, col_min, row_max, col_max = (int(v) for v in bbox)
        bbox_width = col_max - col_min
        bbox_height = row_max - row_min
        _, holes_count, hole_areas = count_holes(component, 8)
        area = int(item.area)
        filled_area = int(area + sum(hole_areas))

        rows.append(
            {
                "label": label_value,
                "area": area,
                "centroid_row": round(float(item.centroid[0]), 4),
                "centroid_col": round(float(item.centroid[1]), 4),
                "centroid_x": round(x_center + 1.0, 4),
                "centroid_y": round(y_center + 1.0, 4),
                "bbox_x": col_min + 1,
                "bbox_y": row_min + 1,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "perimeter": round(float(item.perimeter), 4),
                "eccentricity": round(eccentricity, 4),
                "orientation_deg": round(orientation, 4),
                "major_axis_length": round(major_length, 4),
                "minor_axis_length": round(minor_length, 4),
                "equivalent_diameter": round(
                    float(np.sqrt(4.0 * area / np.pi)), 4
                ),
                "euler_number": int(1 - holes_count),
                "filled_area": filled_area,
            }
        )
    return rows


# рисунки
def save_input_figure(mask, path):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    image_axis(
        axes[0],
        binary_to_gray(mask),
        f"П. 1. Вхідне бінарне зображення\n"
        f"{mask.shape[0]}x{mask.shape[1]}, значення 0/255",
    )
    counts = [int((~mask).sum()), int(mask.sum())]
    axes[1].bar([0, 255], counts, color=["0.4", "0.8"], width=60)
    axes[1].set_title("Розподіл значень пікселів", fontsize=10)
    axes[1].set_xlabel("Значення пікселя")
    axes[1].set_ylabel("Кількість пікселів")
    axes[1].set_xticks([0, 255])
    for value, count_value in zip([0, 255], counts):
        axes[1].text(
            value,
            count_value,
            f"{count_value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("П. 1. Вхідні дані", fontsize=13)
    save_figure(figure, path)


def save_bwarea_figure(
    mask,
    labels,
    count,
    weighted_total,
    diagonal_mask,
    diagonal_area,
    path,
):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    image_axis(
        axes[0, 0],
        binary_to_gray(mask),
        f"Вхідне зображення\nпіксельна площа {int(mask.sum())}",
    )
    pixel_areas = [
        int((labels == index).sum()) for index in range(1, count + 1)
    ]
    weighted_areas = [
        bwarea_weighted(labels == index) for index in range(1, count + 1)
    ]
    positions = np.arange(1, count + 1)
    axes[0, 1].bar(
        positions - 0.2, pixel_areas, width=0.4, label="кількість пікселів"
    )
    axes[0, 1].bar(
        positions + 0.2, weighted_areas, width=0.4, label="bwarea"
    )
    axes[0, 1].set_title(
        f"Площа по об'єктах\nразом: пікселі {int(mask.sum())}, "
        f"bwarea {weighted_total:.1f}",
        fontsize=10,
    )
    axes[0, 1].set_xlabel("Номер об'єкта")
    axes[0, 1].set_ylabel("Площа, пікселі")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(axis="y", alpha=0.2)

    image_axis(
        axes[1, 0],
        binary_to_gray(diagonal_mask),
        "Синтетична діагональ 32x32\n(один піксель завтовшки)",
    )
    axes[1, 1].bar(
        ["кількість\nпікселів", "bwarea"],
        [int(diagonal_mask.sum()), diagonal_area],
        color=["0.5", "tab:orange"],
    )
    axes[1, 1].set_title(
        "bwarea vs пікселі на діагоналі\n"
        f"{int(diagonal_mask.sum())} -> {diagonal_area:.2f} "
        f"(+{diagonal_area - int(diagonal_mask.sum()):.2f})",
        fontsize=10,
    )
    axes[1, 1].set_ylabel("Площа, пікселі")
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "П. 2. bwarea: зважена площа (ваги вікон 2x2)", fontsize=13
    )
    save_figure(figure, path)


def save_bwareaopen_figure(
    mask,
    opened,
    removed_mask,
    kept,
    removed,
    areas,
    path,
):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    image_axis(
        axes[0],
        highlight_mask(mask, removed_mask),
        f"До bwareaopen: {int((areas > 0).sum())} об'єктів\n"
        "червоним виділено дрібні об'єкти",
    )
    image_axis(
        axes[1],
        binary_to_gray(opened),
        f"Після bwareaopen(P={AREA_OPEN_THRESHOLD}): "
        f"{len(kept)} об'єкти\n"
        f"залишилися площі {sorted(int(areas[i - 1]) for i in kept)}",
    )
    removed_text = ", ".join(
        f"№{i} ({int(areas[i - 1])} пкс)" for i in removed
    )
    figure.suptitle(
        "П. 3. bwareaopen: видалення об'єктів менше "
        f"P={AREA_OPEN_THRESHOLD} пікселів\nВидалено: {removed_text}",
        fontsize=12,
    )
    save_figure(figure, path)


def save_bwlabel_figure(labels, count, areas, rows, path):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    image_axis(
        axes[0],
        labels,
        f"bwlabel: 8-зв'язність, {count} об'єктів\n"
        "кольори - номери компонентів",
        cmap="nipy_spectral",
        vmin=0,
        vmax=count + 1,
    )
    # підписуємо номери компонентів
    for row in rows:
        axes[0].text(
            float(row["centroid_col"]),
            float(row["centroid_row"]),
            str(row["label"]),
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.0},
        )
    positions = np.arange(1, count + 1)
    axes[1].bar(positions, areas[1:], color="tab:blue")
    axes[1].set_title("Площі компонентів (пікселі)", fontsize=10)
    axes[1].set_xlabel("Номер об'єкта")
    axes[1].set_ylabel("Площа, пікселі")
    axes[1].set_xticks(positions)
    for position, value in zip(positions, areas[1:]):
        axes[1].text(
            position, int(value), f"{int(value)}", ha="center", va="bottom",
            fontsize=8,
        )
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "П. 4. bwlabel: нумерація зв'язних компонентів", fontsize=13
    )
    save_figure(figure, path)


def save_bwlabeln_figure(volume, labels6, count6, labels26, count26, path):
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
    volume_projection = volume.max(axis=2).astype(np.uint8) * 255
    labels6_projection = labels6.max(axis=2)
    labels26_projection = labels26.max(axis=2)
    image_axis(
        axes[0],
        volume_projection,
        "3-D масив: два куби на діагоналі\n(проєкція вздовж осі 0)",
    )
    image_axis(
        axes[1],
        labels6_projection,
        f"bwlabeln, 6-зв'язність: {count6} компоненти",
        cmap="nipy_spectral",
        vmin=0,
        vmax=count6 + 1,
    )
    image_axis(
        axes[2],
        labels26_projection,
        f"bwlabeln, 26-зв'язність: {count26} компонента",
        cmap="nipy_spectral",
        vmin=0,
        vmax=count26 + 1,
    )
    figure.suptitle(
        "П. 4. bwlabeln: зв'язність у тривимірному масиві", fontsize=13
    )
    save_figure(figure, path)


def save_bwdist_figure(
    mask,
    distance,
    nearest_linear,
    cityblock,
    chessboard,
    path,
):
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))
    image_axis(
        axes[0, 0],
        distance,
        f"Евклідова D: max {distance.max():.2f}, "
        f"середнє {distance.mean():.2f}",
        cmap="viridis",
        vmin=0.0,
        vmax=float(distance.max()),
    )
    image_axis(
        axes[0, 1],
        cityblock,
        f"Cityblock: max {cityblock.max():.0f}, "
        f"середнє {cityblock.mean():.2f}",
        cmap="viridis",
        vmin=0.0,
        vmax=float(cityblock.max()),
    )
    image_axis(
        axes[0, 2],
        chessboard,
        f"Chessboard: max {chessboard.max():.0f}, "
        f"середнє {chessboard.mean():.2f}",
        cmap="viridis",
        vmin=0.0,
        vmax=float(chessboard.max()),
    )
    image_axis(
        axes[1, 0],
        nearest_linear,
        "Мітки L найближчого пікселя\n(1-базові лінійні індекси)",
        cmap="nipy_spectral",
        vmin=0,
        vmax=float(nearest_linear.max()),
    )
    axes[1, 1].hist(distance.ravel(), bins=48, color="0.3")
    axes[1, 1].set_title("Гістограма відстаней D", fontsize=10)
    axes[1, 1].set_xlabel("Відстань, пікселі")
    axes[1, 1].set_ylabel("Кількість пікселів")
    axes[1, 1].grid(alpha=0.2)
    row = min(60, mask.shape[0] - 1)
    axes[1, 2].plot(distance[row, :], color="tab:blue", label="евклідова")
    axes[1, 2].plot(
        cityblock[row, :], color="tab:orange", linestyle="--", label="cityblock"
    )
    axes[1, 2].plot(
        chessboard[row, :], color="tab:green", linestyle=":", label="chessboard"
    )
    axes[1, 2].set_title(f"Профіль відстаней, рядок {row}", fontsize=10)
    axes[1, 2].set_xlabel("Стовпець")
    axes[1, 2].set_ylabel("Відстань, пікселі")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.2)
    figure.suptitle(
        "П. 5. bwdist: трансформація відстаней і мітки найближчого об'єкта",
        fontsize=13,
    )
    save_figure(figure, path)


def save_bweuler_figure(
    mask,
    hole_mask,
    components,
    holes,
    euler,
    hole_areas,
    per_component,
    path,
):
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    image_axis(
        axes[0],
        binary_to_gray(mask),
        f"Вхідне зображення\nоб'єктів: {components}",
    )
    hole_overlay = np.dstack(
        [
            np.where(mask, 80, 255).astype(np.uint8),
            np.where(mask, 80, 255).astype(np.uint8),
            np.where(mask, 80, 255).astype(np.uint8),
        ]
    ).copy()
    hole_overlay[hole_mask] = (220, 0, 0)
    image_axis(
        axes[1],
        hole_overlay,
        f"Отвори (червоні): {holes}\nплощі {hole_areas} пкс",
    )
    labels_values = [int(item["label"]) for item in per_component]
    euler_values = [int(item["euler"]) for item in per_component]
    axes[2].bar(labels_values, euler_values, color="tab:purple")
    axes[2].set_ylim(0.0, 1.5)
    axes[2].set_xticks(labels_values)
    axes[2].set_title(
        f"Число Ейлера: χ = C - H = {components} - {holes} = {euler}",
        fontsize=10,
    )
    axes[2].set_xlabel("Номер об'єкта")
    axes[2].set_ylabel("χ об'єкта (1 або 0)")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("П. 6. bweuler: топологія бінарного зображення", fontsize=13)
    save_figure(figure, path)


def save_bwfill_figure(mask, filled, hole_mask, hole_areas, path):
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    image_axis(axes[0], binary_to_gray(mask), "До bwfill")
    image_axis(axes[1], binary_to_gray(filled), "Після bwfill('holes')")
    image_axis(
        axes[2],
        overlay_changes(mask, filled),
        f"Заповнено: {int(hole_mask.sum())} пікселів\n"
        f"отвори: {hole_areas}",
    )
    figure.suptitle(
        "П. 7. bwfill: заповнення внутрішніх отворів", fontsize=13
    )
    save_figure(figure, path)


def save_bwmorph_basic_figure(results, path):
    operations = ["erode", "dilate", "open", "close", "remove"]
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    image_axis(
        axes[0, 0],
        binary_to_gray(results["input"]),
        f"Вхідне зображення\n{int(results['input'].sum())} пкс",
    )
    for axis, operation in zip(axes.ravel()[1:], operations):
        result = results[operation]
        image_axis(
            axis,
            binary_to_gray(result),
            f"bwmorph('{operation}')\n{int(result.sum())} пкс",
        )
    figure.suptitle(
        "П. 8. bwmorph: erode, dilate, open, close, remove", fontsize=13
    )
    save_figure(figure, path)


def save_bwmorph_skeleton_figure(results, path):
    operations = ["skel", "thin", "bridge", "clean", "fill", "spur"]
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    image_axis(
        axes[0, 0],
        binary_to_gray(results["input"]),
        f"Вхідне зображення\n{int(results['input'].sum())} пкс",
    )
    for axis, operation in zip(axes.ravel()[1:], operations):
        result = results[operation]
        added = int((result & ~results["input"]).sum())
        removed = int((results["input"] & ~result).sum())
        image_axis(
            axis,
            overlay_changes(results["input"], result),
            f"bwmorph('{operation}')\n"
            f"{int(result.sum())} пкс, +{added} / -{removed}",
        )
    figure.suptitle(
        "П. 9. bwmorph: скелет, утоншення та точкові операції", fontsize=13
    )
    save_figure(figure, path)


def save_bwmorph_test_figure(test_mask, test_results, path):
    operations = ["bridge", "clean", "fill", "spur", "majority"]
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    image_axis(
        axes[0, 0],
        binary_to_gray(test_mask),
        f"Тестовий шаблон\n{int(test_mask.sum())} пкс, "
        f"{int(len(np.unique(ndi.label(test_mask, STRUCT8)[0])) - 1)} об'єктів",
    )
    for axis, operation in zip(axes.ravel()[1:], operations):
        result = test_results[operation]
        added = int((result & ~test_mask).sum())
        removed = int((test_mask & ~result).sum())
        image_axis(
            axis,
            overlay_changes(test_mask, result),
            f"bwmorph('{operation}')\n+{added} / -{removed} пкс",
        )
    figure.suptitle(
        "П. 10. bwmorph: точкові операції на синтетичному шаблоні",
        fontsize=13,
    )
    save_figure(figure, path)


def save_bwperim_figure(mask, perimeter8, perimeter4, path):
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    image_axis(
        axes[0],
        highlight_mask(mask, perimeter8),
        f"bwperim(BW, 8): {int(perimeter8.sum())} пкс\n"
        "червоним - межа",
    )
    image_axis(
        axes[1],
        highlight_mask(mask, perimeter4, color=(255, 165, 0)),
        f"bwperim(BW, 4): {int(perimeter4.sum())} пкс\n"
        "оранжевим - межа",
    )
    axes[2].bar(
        ["8-зв'язність", "4-зв'язність"],
        [int(perimeter8.sum()), int(perimeter4.sum())],
        color=["tab:red", "tab:orange"],
    )
    axes[2].set_title("Кількість граничних пікселів", fontsize=10)
    axes[2].set_ylabel("Пікселі")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "П. 11. bwperim: виділення меж об'єктів", fontsize=13
    )
    save_figure(figure, path)


def save_bwselect_figure(
    mask,
    selected,
    seeds,
    selected_labels,
    selected_area,
    path,
):
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    image_axis(
        axes[0],
        highlight_mask(mask, selected, color=(0, 170, 0)),
        "Затравочні точки (червоні кола)\n"
        "виділені об'єкти - зелені",
    )
    for seed in seeds:
        axes[0].plot(
            float(seed["centroid_col"]),
            float(seed["centroid_row"]),
            marker="o",
            markersize=11,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.8,
        )
        axes[0].plot(
            float(seed["seed_col"]),
            float(seed["seed_row"]),
            marker="x",
            markersize=9,
            color="white",
            markeredgewidth=2.0,
        )
        axes[0].text(
            float(seed["seed_col"]) + 4.0,
            float(seed["seed_row"]) - 4.0,
            f"№{seed['label']}",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.0},
        )
    image_axis(
        axes[1],
        binary_to_gray(selected),
        f"Результат bwselect\nоб'єкти {selected_labels}, "
        f"площа {selected_area} пкс",
    )
    figure.suptitle(
        "П. 12. bwselect: вибір об'єктів за центрами мас", fontsize=13
    )
    save_figure(figure, path)


def save_bwulterode_figure(mask, ultimate, edt_maxima, max_iterations, path):
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.5))
    image_axis(axes[0, 0], binary_to_gray(mask), "Вхідне зображення")

    eroded = mask.copy()
    panels = [(3, axes[0, 1]), (6, axes[0, 2]), (10, axes[1, 0])]
    current_iteration = 0
    for target, axis in panels:
        while current_iteration < target:
            eroded = ndi.binary_erosion(
                eroded, structure=STRUCT8, border_value=0
            )
            current_iteration += 1
        image_axis(
            axis,
            binary_to_gray(eroded),
            f"Ерозія x{target}\n{int(eroded.sum())} пкс",
        )

    image_axis(
        axes[1, 1],
        highlight_mask(mask, ultimate, color=(220, 0, 0)),
        f"Гранична ерозія (ітеративна)\n"
        f"{int(ultimate.sum())} пкс, max {max_iterations} ітерацій",
    )
    image_axis(
        axes[1, 2],
        highlight_mask(mask, edt_maxima, color=(0, 80, 220)),
        f"Локальні максимуми EDT\n{int(edt_maxima.sum())} пкс "
        "(довідково, як у matlab)",
    )
    figure.suptitle(
        "П. 13. bwulterode: гранична ерозія об'єктів", fontsize=13
    )
    save_figure(figure, path)


def save_regionprops_overlay_figure(mask, rows, path):
    figure, axis = plt.subplots(figsize=(7.8, 7.8))
    axis.imshow(
        binary_to_gray(mask),
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(rows), 2)))
    for index, row in enumerate(rows):
        col_min = int(row["bbox_x"]) - 1
        row_min = int(row["bbox_y"]) - 1
        rectangle = Rectangle(
            (col_min, row_min),
            int(row["bbox_width"]),
            int(row["bbox_height"]),
            fill=False,
            edgecolor=colors[index % len(colors)],
            linewidth=1.6,
        )
        axis.add_patch(rectangle)
        center_col = float(row["centroid_col"])
        center_row = float(row["centroid_row"])
        axis.plot(
            center_col,
            center_row,
            marker="+",
            markersize=10,
            markeredgewidth=2.0,
            color="red",
        )
        angle = np.radians(float(row["orientation_deg"]))
        half_length = max(float(row["major_axis_length"]) * 0.45, 4.0)
        delta_col = half_length * np.cos(angle)
        delta_row = half_length * np.sin(angle)
        axis.plot(
            [center_col - delta_col, center_col + delta_col],
            [center_row - delta_row, center_row + delta_row],
            color="red",
            linewidth=1.2,
        )
        axis.text(
            center_col + 3.0,
            center_row - 3.0,
            f"№{row['label']}",
            color="yellow",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.0},
        )
    axis.set_title(
        "П. 14. regionprops: рамки, центри мас і головні осі\n"
        f"об'єктів: {len(rows)}",
        fontsize=12,
    )
    axis.set_xlabel("x (стовпці)")
    axis.set_ylabel("y (рядки)")
    save_figure(figure, path)


def save_regionprops_table_figure(rows, path):
    columns = [
        "№",
        "Площа,\nпкс",
        "Центр (x; y)",
        "Рамка [x y w h]",
        "Периметр,\nпкс",
        "Ексцен-\nтриситет",
        "Орієнта-\nція, °",
    ]
    cell_text = []
    for row in rows:
        cell_text.append(
            [
                str(row["label"]),
                str(row["area"]),
                f"{float(row['centroid_x']):.1f}; "
                f"{float(row['centroid_y']):.1f}",
                f"{row['bbox_x']} {row['bbox_y']} "
                f"{row['bbox_width']} {row['bbox_height']}",
                f"{float(row['perimeter']):.1f}",
                f"{float(row['eccentricity']):.3f}",
                f"{float(row['orientation_deg']):.1f}",
            ]
        )
    figure, axis = plt.subplots(figsize=(14.0, 4.5))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.8)
    axis.set_title(
        "П. 15. regionprops: параметри об'єктів",
        fontsize=12,
        pad=18,
    )
    save_figure(figure, path)


# синтетичні зображення
def make_diagonal_demo():
    # діагональ 32x32 для демонстрації зважування bwarea
    return np.eye(32, dtype=bool)


def make_morphology_test_pattern():
    # шаблон для точкових операцій bridge, clean, fill, spur
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:21, 8:21] = True       # лівий прямокутник
    mask[8:21, 22:35] = True      # правий прямокутник, розрив 1 піксель
    mask[26:29, 26:29] = True     # квадрат 3x3 з отвором у центрі
    mask[27, 27] = False
    mask[35, 35] = True           # ізольований піксель
    mask[45, 5:25] = True         # лінія з двома кінцями (spur)
    return mask


def make_3d_volume():
    # два куби, що торкаються лише кутом
    volume = np.zeros((26, 26, 26), dtype=bool)
    volume[4:13, 4:13, 4:13] = True
    volume[13:22, 13:22, 13:22] = True
    return volume


# основний сценарій
def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # читаємо маску
    input_image = load_gray(INPUT_PATH)
    mask = input_image > 0

    # мітки компонентів
    labels8, components8 = bwlabel_matlab(mask, 8)
    labels4, components4 = bwlabel_matlab(mask, 4)
    _, componentsn8 = bwlabeln_matlab(mask, 8)
    _, componentsn4 = bwlabeln_matlab(mask, 4)
    areas = np.bincount(labels8.ravel(), minlength=components8 + 1)
    rows = build_regionprops(labels8, components8)
    hole_mask, holes_count, hole_areas = count_holes(mask, 8)
    euler, euler_components, euler_holes = bweuler_matlab(mask)

    print("Практична робота 6: вимір параметрів графічних об'єктів")
    print(f"Вхідний файл: {INPUT_PATH}")
    print(
        f"Розмір: {mask.shape[0]}x{mask.shape[1]} (рядки x стовпці), "
        f"значення: {[int(v) for v in np.unique(input_image)]}"
    )
    print(
        f"Передній план: {int(mask.sum())} пкс "
        f"({100.0 * mask.mean():.1f}%), фон: {int((~mask).sum())} пкс"
    )
    print(
        f"Компоненти: 8-зв'язність - {components8}, "
        f"4-зв'язність - {components4}; отворів: {holes_count}; χ = {euler}"
    )

    print("\nП. 1. вхідне зображення")
    save_input_figure(mask, OUT_DIR / FIGURE_FILES[0])

    print("\nП. 2. bwarea")
    weighted_area = bwarea_weighted(mask)
    diagonal_demo = make_diagonal_demo()
    diagonal_area = bwarea_weighted(diagonal_demo)
    save_bwarea_figure(
        mask,
        labels8,
        components8,
        weighted_area,
        diagonal_demo,
        diagonal_area,
        OUT_DIR / FIGURE_FILES[1],
    )
    print(f"Піксельна площа: {int(mask.sum())}")
    print(f"bwarea (зважена): {weighted_area:.2f}")
    print(f"Різниця: {weighted_area - int(mask.sum()):.2f}")
    print(
        "Діагональ 32x32: "
        f"{int(diagonal_demo.sum())} пкс, bwarea {diagonal_area:.2f}"
    )

    print("\nП. 3. bwareaopen")
    opened, _, kept, removed, open_areas = bwareaopen_matlab(
        mask, AREA_OPEN_THRESHOLD, 8
    )
    removed_mask = mask & ~opened
    save_gray(OUT_DIR / "bwareaopen.png", binary_to_gray(opened))
    save_bwareaopen_figure(
        mask,
        opened,
        removed_mask,
        kept,
        removed,
        open_areas,
        OUT_DIR / FIGURE_FILES[2],
    )
    print(
        f"Компонентів до: {components8}, після: {len(kept)}; "
        f"видалено {len(removed)}"
    )
    print("Видалені: " + ", ".join(
        f"№{i} ({int(open_areas[i - 1])} пкс)" for i in removed
    ))
    print(f"Сумарно видалено пікселів: {int(removed_mask.sum())}")

    print("\nП. 4. bwlabel / bwlabeln")
    save_bwlabel_figure(
        labels8, components8, areas, rows, OUT_DIR / FIGURE_FILES[3]
    )
    volume = make_3d_volume()
    labels_3d_6, count_3d_6 = bwlabeln_matlab(volume, 6)
    labels_3d_26, count_3d_26 = bwlabeln_matlab(volume, 26)
    save_bwlabeln_figure(
        volume,
        labels_3d_6,
        count_3d_6,
        labels_3d_26,
        count_3d_26,
        OUT_DIR / FIGURE_FILES[4],
    )
    print(f"bwlabel, 8-зв'язність: {components8} компонентів")
    print(f"bwlabel, 4-зв'язність: {components4} компонентів")
    print(
        f"bwlabeln, 2-D: 8-зв'язність - {componentsn8}, "
        f"4-зв'язність - {componentsn4}"
    )
    print(
        f"bwlabeln, 3-D приклад: 6-зв'язність - {count_3d_6}, "
        f"26-зв'язність - {count_3d_26}"
    )

    print("\nП. 5. bwdist")
    distance, nearest_linear, nearest_rows, nearest_cols = bwdist_matlab(mask)
    cityblock = ndi.distance_transform_cdt(~mask, metric="taxicab")
    chessboard = ndi.distance_transform_cdt(~mask, metric="chessboard")
    save_bwdist_figure(
        mask,
        distance,
        nearest_linear,
        cityblock,
        chessboard,
        OUT_DIR / FIGURE_FILES[5],
    )
    print(
        f"Евклідова D: max {distance.max():.2f}, "
        f"середнє {distance.mean():.2f}"
    )
    print(
        f"Cityblock: max {cityblock.max():.0f}; "
        f"chessboard: max {chessboard.max():.0f}"
    )
    print(
        "Мітки L: 1-базові лінійні індекси, діапазон "
        f"{int(nearest_linear.min())}..{int(nearest_linear.max())}"
    )

    print("\nП. 6. bweuler")
    per_component_euler = []
    for index in range(1, components8 + 1):
        _, component_holes, _ = count_holes(labels8 == index, 8)
        per_component_euler.append(
            {
                "label": index,
                "holes": int(component_holes),
                "euler": int(1 - component_holes),
            }
        )
    save_bweuler_figure(
        mask,
        hole_mask,
        components8,
        holes_count,
        euler,
        hole_areas,
        per_component_euler,
        OUT_DIR / FIGURE_FILES[6],
    )
    print(f"C = {euler_components}, H = {euler_holes}, χ = {euler}")
    print(
        "Покомпонентно: "
        + ", ".join(
            f"№{item['label']} - χ={item['euler']}"
            for item in per_component_euler
        )
    )

    print("\nП. 7. bwfill")
    filled, fill_hole_mask, fill_hole_areas = bwfill_holes(mask, 8)
    _, holes_after_fill, _ = count_holes(filled, 8)
    save_gray(OUT_DIR / "bwfill.png", binary_to_gray(filled))
    save_bwfill_figure(
        mask,
        filled,
        fill_hole_mask,
        fill_hole_areas,
        OUT_DIR / FIGURE_FILES[7],
    )
    print(
        f"Отворів до: {holes_count} (площі {hole_areas}), "
        f"заповнено {int(fill_hole_mask.sum())} пкс, "
        f"отворів після: {holes_after_fill}"
    )

    print("\nП. 8. bwmorph")
    morph_results = {"input": mask}
    for operation in MORPH_MAIN_OPERATIONS:
        morph_results[operation] = bwmorph_matlab(mask, operation)
    morph_results["skel"] = bwmorph_matlab(mask, "skel")
    morph_results["thin"] = bwmorph_matlab(mask, "thin")
    save_gray(OUT_DIR / "skel.png", binary_to_gray(morph_results["skel"]))
    save_gray(OUT_DIR / "thin.png", binary_to_gray(morph_results["thin"]))
    save_bwmorph_basic_figure(
        morph_results, OUT_DIR / FIGURE_FILES[8]
    )
    save_bwmorph_skeleton_figure(
        morph_results, OUT_DIR / FIGURE_FILES[9]
    )
    morph_counts = {
        operation: int(morph_results[operation].sum())
        for operation in MORPH_SUPPORTED
    }
    print(
        f"Підтримано: {', '.join(MORPH_SUPPORTED)}; "
        f"не підтримано: {', '.join(MORPH_UNSUPPORTED)}"
    )
    print(
        "Площі після операцій: "
        + ", ".join(
            f"{operation}={morph_counts[operation]}"
            for operation in MORPH_SUPPORTED
        )
    )

    test_mask = make_morphology_test_pattern()
    test_results = {
        operation: bwmorph_matlab(test_mask, operation)
        for operation in ["bridge", "clean", "fill", "spur", "majority"]
    }
    test_records = {}
    for operation, result in test_results.items():
        test_records[operation] = {
            "added": int((result & ~test_mask).sum()),
            "removed": int((test_mask & ~result).sum()),
        }
    save_gray(OUT_DIR / "bwmorph_test.png", binary_to_gray(test_mask))
    save_bwmorph_test_figure(
        test_mask, test_results, OUT_DIR / FIGURE_FILES[10]
    )
    print(
        "Тестовий шаблон: "
        + ", ".join(
            f"{operation} (+{record['added']}/-{record['removed']})"
            for operation, record in test_records.items()
        )
    )

    print("\nП. 9. bwperim")
    perimeter8 = bwperim_matlab(mask, 8)
    perimeter4 = bwperim_matlab(mask, 4)
    save_gray(OUT_DIR / "bwperim8.png", binary_to_gray(perimeter8))
    save_gray(OUT_DIR / "bwperim4.png", binary_to_gray(perimeter4))
    save_bwperim_figure(
        mask, perimeter8, perimeter4, OUT_DIR / FIGURE_FILES[11]
    )
    print(
        f"8-зв'язність: {int(perimeter8.sum())} граничних пікселів; "
        f"4-зв'язність: {int(perimeter4.sum())}"
    )

    print("\nП. 10. bwselect")
    # затравочні точки беремо у двох найбільших об'єктів
    largest_labels = sorted(
        range(1, components8 + 1),
        key=lambda index: int(areas[index]),
        reverse=True,
    )[:2]
    seeds = []
    for label_value in largest_labels:
        component = labels8 == label_value
        record = next(item for item in rows if item["label"] == label_value)
        centroid_row = float(record["centroid_row"])
        centroid_col = float(record["centroid_col"])
        seed_row, seed_col = snap_seed_to_foreground(
            component, centroid_row, centroid_col
        )
        seeds.append(
            {
                "label": int(label_value),
                "centroid_row": round(centroid_row, 4),
                "centroid_col": round(centroid_col, 4),
                "seed_row": int(seed_row),
                "seed_col": int(seed_col),
            }
        )
    selected, selected_labels = bwselect_matlab(
        labels8, [(item["seed_row"], item["seed_col"]) for item in seeds]
    )
    selected_area = int(selected.sum())
    save_gray(OUT_DIR / "bwselect.png", binary_to_gray(selected))
    save_bwselect_figure(
        mask,
        selected,
        seeds,
        selected_labels,
        selected_area,
        OUT_DIR / FIGURE_FILES[12],
    )
    print(
        "Затравочні точки: "
        + ", ".join(
            f"№{item['label']} ({item['seed_row']},{item['seed_col']})"
            for item in seeds
        )
    )
    print(
        f"Виділено об'єкти {selected_labels}, "
        f"площа {selected_area} пкс"
    )

    print("\nП. 11. bwulterode")
    ultimate, ultimate_records, max_iterations = bwulterode_iterative(mask)
    edt_maxima, _ = bwulterode_edt(mask)
    save_gray(OUT_DIR / "bwulterode.png", binary_to_gray(ultimate))
    save_bwulterode_figure(
        mask,
        ultimate,
        edt_maxima,
        max_iterations,
        OUT_DIR / FIGURE_FILES[13],
    )
    ultimate_components = int(ndi.label(ultimate, structure=STRUCT8)[1])
    print(
        f"Ітеративна гранична ерозія: {int(ultimate.sum())} пкс, "
        f"{ultimate_components} кластерів, max {max_iterations} ітерацій"
    )
    print(f"Локальні максимуми EDT: {int(edt_maxima.sum())} пкс")
    print(
        "По компонентах: "
        + ", ".join(
            f"№{item['label']} - {item['points']} пкс "
            f"(ітерацій: {item['iterations']})"
            for item in ultimate_records
        )
    )

    print("\nП. 12. regionprops")
    save_regionprops_overlay_figure(
        mask, rows, OUT_DIR / FIGURE_FILES[14]
    )
    save_regionprops_table_figure(rows, OUT_DIR / FIGURE_FILES[15])
    header = (
        "№   Площа  Центр(x;y)      Рамка[x y w h]    "
        "Перим.  Ексц.  Орієнт."
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:>2}  {row['area']:>6}  "
            f"({float(row['centroid_x']):6.1f};{float(row['centroid_y']):6.1f})  "
            f"[{row['bbox_x']:>3} {row['bbox_y']:>3} "
            f"{row['bbox_width']:>3} {row['bbox_height']:>3}]  "
            f"{float(row['perimeter']):>6.1f}  "
            f"{float(row['eccentricity']):>5.3f}  "
            f"{float(row['orientation_deg']):>6.1f}"
        )

    # детальна таблиця у csv
    csv_path = OUT_DIR / "regionprops.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Таблицю збережено: {csv_path}")

    results = {
        "title": "вимір параметрів графічних об'єктів",
        "parameters": {
            "area_open_threshold": AREA_OPEN_THRESHOLD,
            "connectivity": 8,
            "background_connectivity": 4,
        },
        "input": {
            "file": "data/binary_shapes.png",
            "shape": [int(mask.shape[0]), int(mask.shape[1])],
            "foreground_pixels": int(mask.sum()),
            "background_pixels": int((~mask).sum()),
            "components_8": int(components8),
            "components_4": int(components4),
            "holes": int(holes_count),
            "hole_areas": [int(value) for value in hole_areas],
            "euler_number": int(euler),
        },
        "bwarea": {
            "pixel_area": int(mask.sum()),
            "weighted_area": round(weighted_area, 4),
            "difference": round(weighted_area - int(mask.sum()), 4),
            "diagonal_pixels": int(diagonal_demo.sum()),
            "diagonal_weighted_area": round(diagonal_area, 4),
            "figure": "out/02_bwarea.png",
        },
        "bwareaopen": {
            "threshold": AREA_OPEN_THRESHOLD,
            "components_before": int(components8),
            "components_after": int(len(kept)),
            "kept_labels": [int(value) for value in kept],
            "removed_labels": [int(value) for value in removed],
            "removed_pixels": int(removed_mask.sum()),
            "mask": "out/bwareaopen.png",
            "figure": "out/03_bwareaopen.png",
        },
        "bwlabel": {
            "components_8": int(components8),
            "components_4": int(components4),
            "bwlabeln_2d": {
                "components_8": int(componentsn8),
                "components_4": int(componentsn4),
            },
            "bwlabeln_3d": {
                "volume_shape": [int(value) for value in volume.shape],
                "components_6": int(count_3d_6),
                "components_26": int(count_3d_26),
            },
            "figures": ["out/04_bwlabel.png", "out/05_bwlabeln_3d.png"],
        },
        "bwdist": {
            "metric": "euclidean",
            "max_distance": round(float(distance.max()), 4),
            "mean_distance": round(float(distance.mean()), 4),
            "cityblock_max": round(float(cityblock.max()), 4),
            "chessboard_max": round(float(chessboard.max()), 4),
            "figure": "out/06_bwdist.png",
        },
        "bweuler": {
            "euler_number": int(euler),
            "components": int(euler_components),
            "holes": int(euler_holes),
            "figure": "out/07_bweuler.png",
        },
        "bwfill": {
            "holes_before": int(holes_count),
            "hole_areas": [int(value) for value in fill_hole_areas],
            "filled_pixels": int(fill_hole_mask.sum()),
            "holes_after": int(holes_after_fill),
            "mask": "out/bwfill.png",
            "figure": "out/08_bwfill.png",
        },
        "bwmorph": {
            "supported": list(MORPH_SUPPORTED),
            "counts": {
                operation: int(morph_counts[operation])
                for operation in MORPH_SUPPORTED
            },
            "test_pattern": {
                "mask": "out/bwmorph_test.png",
                "operations": test_records,
            },
            "figures": [
                "out/09_bwmorph_basic.png",
                "out/10_bwmorph_skeleton.png",
                "out/11_bwmorph_test_pattern.png",
            ],
        },
        "bwperim": {
            "pixels_8": int(perimeter8.sum()),
            "pixels_4": int(perimeter4.sum()),
            "masks": ["out/bwperim8.png", "out/bwperim4.png"],
            "figure": "out/12_bwperim.png",
        },
        "bwselect": {
            "seeds": [
                {
                    "label": item["label"],
                    "seed_row": item["seed_row"],
                    "seed_col": item["seed_col"],
                }
                for item in seeds
            ],
            "selected_labels": [int(value) for value in selected_labels],
            "selected_area": int(selected_area),
            "mask": "out/bwselect.png",
            "figure": "out/13_bwselect.png",
        },
        "bwulterode": {
            "iterative_points": int(ultimate.sum()),
            "iterative_components": int(ultimate_components),
            "max_iterations": int(max_iterations),
            "edt_maxima_points": int(edt_maxima.sum()),
            "mask": "out/bwulterode.png",
            "figure": "out/14_bwulterode.png",
        },
        "regionprops": {
            "csv": "out/regionprops.csv",
            "count": len(rows),
            "total_area": int(sum(int(item["area"]) for item in rows)),
            "mean_area": round(
                float(np.mean([int(item["area"]) for item in rows])), 4
            ),
            "figures": [
                "out/15_regionprops_overlay.png",
                "out/16_regionprops_table.png",
            ],
        },
        "figures": ["out/%s" % name for name in FIGURE_FILES],
        "masks": ["out/%s" % name for name in MASK_FILES],
    }

    results_path = OUT_DIR / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nЗбережено JSON: {results_path}")
    print("Готово")
    return 0


def setup_console():
    # щоб кирилиця не ламалась у консолі
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main():
    setup_console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
