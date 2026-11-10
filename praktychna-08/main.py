# практична робота 8. пошук кіл і сегментація кольорового зображення
# запуск: python main.py

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2lab

from utils import image_axis, load_rgb, save_figure, save_rgb

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"

SEED = 42

# параметри детектора кіл
HOUGH_PARAMS = {
    "dp": 1.2,
    "minDist": 20,
    "param1": 120,
    "param2": 45,
    "minRadius": 10,
    "maxRadius": 60,
    "blur_kernel": 5,
    "blur_sigma": 1.5,
}

# параметри кластеризації k-середніх
KMEANS_PARAMS = {
    "k": 3,
    "attempts": 10,
    "max_iter": 100,
    "epsilon": 0.2,
}

# кольори кластерів для рисунків
CLUSTER_PALETTE = np.array(
    [
        [228, 26, 28],
        [55, 126, 184],
        [77, 175, 74],
        [255, 127, 0],
        [152, 78, 163],
    ],
    dtype=np.uint8,
)


def setup_console():
    # щоб український текст не ламався у вікні термінала
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def relative_path(path):
    # шлях відносно папки практичної роботи
    return path.relative_to(BASE_DIR).as_posix()


def figure_size(shape, max_width=10.0):
    # підбираємо розмір рисунка під розмір зображення
    height, width = shape[:2]
    figure_width = min(max_width, max(5.0, width / 70.0 + 1.5))
    figure_height = figure_width * height / width + 1.5
    return figure_width, figure_height


def save_image_figure(image, title, path, cmap="gray"):
    # зберігаємо картинку окремим рисунком з підписом
    figure, axis = plt.subplots(figsize=figure_size(image.shape))
    image_axis(axis, image, title, cmap=cmap)
    save_figure(figure, path)


def preprocess_gray(gray, params):
    # згладжуємо сіре зображення гаусовим фільтром
    kernel = int(params["blur_kernel"])
    sigma = float(params["blur_sigma"])
    return cv2.GaussianBlur(gray, (kernel, kernel), sigma)


def gradient_magnitude(gray):
    # модуль градієнта, оператор собеля
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def sample_circle(image, center_x, center_y, radius, angles=360):
    # беремо значення вздовж кола заданого радіуса
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    map_x = (center_x + radius * np.cos(theta)).astype(np.float32).reshape(1, -1)
    map_y = (center_y + radius * np.sin(theta)).astype(np.float32).reshape(1, -1)
    sampled = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.ravel()


def circle_edge_support(magnitude, center_x, center_y, radius):
    # наскільки контур кола підкріплений градієнтами, від 0 до 1
    best = 0.0
    for offset in (-1.0, 0.0, 1.0):
        values = sample_circle(magnitude, center_x, center_y, radius + offset)
        best = max(best, float(values.mean()))
    scale = float(np.percentile(magnitude, 99.0))
    if scale <= 0.0:
        return 0.0
    return float(min(best / scale, 1.0))


def circle_intensity_contrast(gray, center_x, center_y, radius):
    # різниця яскравості між центром кола і фоном
    inside = float(sample_circle(gray, center_x, center_y, 0.5 * radius).mean())
    outside = float(sample_circle(gray, center_x, center_y, 1.4 * radius).mean())
    return float(abs(inside - outside))


def hough_circles(blurred, params):
    # перетворення хафа для кіл
    cv2.setRNGSeed(SEED)
    found = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=float(params["dp"]),
        minDist=float(params["minDist"]),
        param1=float(params["param1"]),
        param2=float(params["param2"]),
        minRadius=int(params["minRadius"]),
        maxRadius=int(params["maxRadius"]),
    )
    if found is None:
        return np.empty((0, 3), dtype=np.float32)
    circles = np.asarray(found[0], dtype=np.float32)
    # сортуємо кола по координатах
    order = np.lexsort((circles[:, 0], circles[:, 1]))
    return circles[order]


def detect_circles(gray, params):
    # згладжуємо і шукаємо кола
    return hough_circles(preprocess_gray(gray, params), params)


def radius_statistics(radii):
    # статистики по радіусах знайдених кіл
    radii = np.asarray(radii, dtype=np.float64)
    return {
        "min": float(radii.min()),
        "max": float(radii.max()),
        "mean": float(radii.mean()),
        "std": float(radii.std(ddof=0)),
        "median": float(np.median(radii)),
    }


def study_sensitivity(gray, base_params):
    # дивимось як param1, param2 і mindist впливають на кількість кіл
    blurred = preprocess_gray(gray, base_params)
    param2_values = list(range(20, 66, 5))
    param1_values = [80, 100, 120, 160]
    counts_by_param1 = {}

    for param1 in param1_values:
        counts = []
        for param2 in param2_values:
            params = dict(base_params)
            params["param1"] = param1
            params["param2"] = param2
            counts.append(int(len(hough_circles(blurred, params))))
        counts_by_param1[param1] = counts

    min_dist_values = [12, 16, 20, 24, 28, 32, 36, 40, 45, 50]
    min_dist_counts = []
    for min_dist in min_dist_values:
        params = dict(base_params)
        params["minDist"] = min_dist
        min_dist_counts.append(int(len(hough_circles(blurred, params))))

    # два графіки: по param2 і по mindist
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    markers = ["o", "s", "^", "D"]
    for index, param1 in enumerate(param1_values):
        axes[0].plot(
            param2_values,
            counts_by_param1[param1],
            marker=markers[index % len(markers)],
            linewidth=1.5,
            label=f"param1 = {param1}",
        )
    axes[0].axvline(
        float(base_params["param2"]),
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label=f"робоче param2 = {base_params['param2']}",
    )
    axes[0].set_title("Вплив param2 на кількість кіл")
    axes[0].set_xlabel("param2, менше значення - більше кіл")
    axes[0].set_ylabel("кількість кіл")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        min_dist_values,
        min_dist_counts,
        marker="o",
        color="tab:orange",
        linewidth=1.5,
    )
    axes[1].axvline(
        float(base_params["minDist"]),
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label=f"робоче minDist = {base_params['minDist']}",
    )
    axes[1].set_title("Вплив minDist на кількість кіл")
    axes[1].set_xlabel("minDist, пікселі")
    axes[1].set_ylabel("кількість кіл")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Дослідження параметрів детектора кіл", fontsize=12)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "circles_sensitivity.png")

    # друкуємо таблицю кількості кіл
    print("\nкількість кіл по param1 і param2:")
    print("param1\\param2 " + " ".join(f"{value:>4}" for value in param2_values))
    for param1 in param1_values:
        print(
            f"{param1:>13} "
            + " ".join(f"{count:>4}" for count in counts_by_param1[param1])
        )
    print("\nвплив minDist:")
    print(
        "  "
        + ", ".join(
            f"minDist={value}: {count}"
            for value, count in zip(min_dist_values, min_dist_counts)
        )
    )


def draw_circles(rgb, circles):
    # малюємо кола, центри і номери на копії зображення
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for index, (center_x, center_y, radius) in enumerate(circles, start=1):
        center = (int(round(float(center_x))), int(round(float(center_y))))
        cv2.circle(canvas, center, int(round(float(radius))), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def run_circle_analysis(rgb):
    # шукаємо кола і зберігаємо рисунки та таблицю
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = preprocess_gray(gray, HOUGH_PARAMS)
    canny_low = int(HOUGH_PARAMS["param1"]) // 2
    canny_high = int(HOUGH_PARAMS["param1"])
    edges = cv2.Canny(blurred, canny_low, canny_high)
    magnitude = gradient_magnitude(blurred)

    circles = detect_circles(gray, HOUGH_PARAMS)
    if len(circles) == 0:
        raise RuntimeError("не знайдено жодного кола, перевір параметри")

    rows = []
    for index, (center_x, center_y, radius) in enumerate(circles, start=1):
        rows.append(
            {
                "id": index,
                "center_x": float(center_x),
                "center_y": float(center_y),
                "radius": float(radius),
                "edge_support": circle_edge_support(magnitude, center_x, center_y, radius),
                "intensity_contrast": circle_intensity_contrast(
                    blurred, center_x, center_y, radius
                ),
            }
        )

    radii = np.array([row["radius"] for row in rows], dtype=np.float64)
    stats = radius_statistics(radii)

    # оригінал, сіра версія і підготовка
    save_image_figure(rgb, "Вхідне зображення з колами", OUT_DIR / "circles_original.png")
    save_image_figure(gray, "Напівтонове зображення", OUT_DIR / "circles_gray.png")

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    image_axis(axes[0], gray, "градації сірого")
    image_axis(
        axes[1],
        blurred,
        f"згладжування {int(HOUGH_PARAMS['blur_kernel'])}x{int(HOUGH_PARAMS['blur_kernel'])}",
    )
    image_axis(axes[2], edges, f"контури canny ({canny_low}, {canny_high})")
    figure.suptitle("Підготовка зображення до пошуку кіл", fontsize=12)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "circles_preprocessed.png")

    # картинка з колами
    annotated = draw_circles(rgb, circles)
    save_rgb(OUT_DIR / "circles_annotated.png", annotated)
    figure, axis = plt.subplots(figsize=figure_size(rgb.shape, max_width=10.5))
    image_axis(
        axis,
        annotated,
        f"знайдено кіл: {len(rows)} (dp={HOUGH_PARAMS['dp']}, "
        f"minDist={HOUGH_PARAMS['minDist']}, param2={HOUGH_PARAMS['param2']})",
    )
    save_figure(figure, OUT_DIR / "circles_hough_overlay.png")

    # центри і радіуси кіл
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].imshow(rgb, alpha=0.35)
    scatter = axes[0].scatter(
        circles[:, 0],
        circles[:, 1],
        c=radii,
        cmap="viridis",
        s=40,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )
    axes[0].set_title("Центри кіл, колір - радіус")
    axes[0].set_xlabel("x, пікселі")
    axes[0].set_ylabel("y, пікселі")
    figure.colorbar(scatter, ax=axes[0], label="Радіус, пікселі")

    axes[1].hist(
        radii,
        bins=np.arange(stats["min"] - 1.5, stats["max"] + 2.0, 1.0),
        color="0.35",
        edgecolor="white",
    )
    axes[1].axvline(
        stats["mean"], color="tab:red", linestyle="--", label=f"середнє = {stats['mean']:.2f}"
    )
    axes[1].set_title("Розподіл радіусів")
    axes[1].set_xlabel("Радіус, пікселі")
    axes[1].set_ylabel("Кількість кіл")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "circles_center_radius.png")

    # таблиця з координатами кіл
    with (OUT_DIR / "circles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["id", "center_x", "center_y", "radius", "edge_support", "intensity_contrast"]
        )
        for row in rows:
            writer.writerow(
                [
                    row["id"],
                    f"{row['center_x']:.3f}",
                    f"{row['center_y']:.3f}",
                    f"{row['radius']:.3f}",
                    f"{row['edge_support']:.4f}",
                    f"{row['intensity_contrast']:.3f}",
                ]
            )

    # дивимось на чутливість детектора
    study_sensitivity(gray, HOUGH_PARAMS)

    print("\nзнайдені кола (x, y, r, підтримка контуру, перепад яскравості):")
    print(f"{'№':>3} {'x':>8} {'y':>8} {'r':>7} {'support':>8} {'contrast':>9}")
    for row in rows:
        print(
            f"{row['id']:>3} {row['center_x']:>8.2f} {row['center_y']:>8.2f} "
            f"{row['radius']:>7.2f} {row['edge_support']:>8.3f} {row['intensity_contrast']:>9.2f}"
        )
    print(
        f"\nкіл: {len(rows)}, радіус від {stats['min']:.2f} до {stats['max']:.2f}, "
        f"середнє {stats['mean']:.2f}, std {stats['std']:.2f}, медіана {stats['median']:.2f}"
    )

    return {"count": len(rows), "stats": stats, "rows": rows}


def select_cluster(centers):
    # беремо кластер з найбільшою насиченістю
    chroma = np.linalg.norm(centers, axis=1)
    return int(np.argmax(chroma))


def segment_colors(rgb):
    # кластеризуємо пікселі за кольоровістю a* і b*
    k = int(KMEANS_PARAMS["k"])
    lab = rgb2lab(rgb)
    ab = lab[:, :, 1:3].reshape(-1, 2).astype(np.float32)

    cv2.setRNGSeed(SEED)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(KMEANS_PARAMS["max_iter"]),
        float(KMEANS_PARAMS["epsilon"]),
    )
    compactness, raw_labels, raw_centers = cv2.kmeans(
        ab,
        k,
        None,
        criteria,
        int(KMEANS_PARAMS["attempts"]),
        cv2.KMEANS_PP_CENTERS,
    )
    labels = raw_labels.ravel().reshape(rgb.shape[:2])
    centers = np.asarray(raw_centers, dtype=np.float64)

    # переставляємо кластери за кількістю пікселів, фоновий буде першим
    raw_counts = np.bincount(labels.ravel(), minlength=k).astype(np.int64)
    order = np.argsort(-raw_counts)
    old_to_new = np.empty(k, dtype=np.int64)
    for new_index, old_index in enumerate(order):
        old_to_new[old_index] = new_index
    labels = old_to_new[labels]
    centers = centers[order]
    counts = raw_counts[order]

    height, width = labels.shape
    total = int(height * width)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float64)
    flat_l = lab[:, :, 0].ravel()
    chroma = np.linalg.norm(centers, axis=1)

    clusters = []
    for index in range(k):
        mask = labels.ravel() == index
        clusters.append(
            {
                "cluster": index + 1,
                "center_a": float(centers[index, 0]),
                "center_b": float(centers[index, 1]),
                "chroma": float(chroma[index]),
                "pixel_count": int(counts[index]),
                "pixel_share": float(counts[index] / total),
                "mean_rgb": [float(value) for value in flat_rgb[mask].mean(axis=0)],
                "mean_l": float(flat_l[mask].mean()),
            }
        )

    selected = select_cluster(centers)
    for index, info in enumerate(clusters):
        info["selected"] = bool(index == selected)

    return {
        "lab": lab,
        "labels": labels,
        "centers": centers,
        "counts": counts,
        "compactness": float(compactness),
        "clusters": clusters,
        "selected": selected,
        "total_pixels": total,
    }


def run_segmentation(rgb):
    # сегментуємо зображення і зберігаємо рисунки
    k = int(KMEANS_PARAMS["k"])
    result = segment_colors(rgb)
    labels = result["labels"]
    centers = result["centers"]
    counts = result["counts"]
    clusters = result["clusters"]
    selected = result["selected"]
    total = result["total_pixels"]

    save_image_figure(
        rgb,
        "Вхідне зображення до сегментації",
        OUT_DIR / "segmentation_original.png",
    )

    # карта кластерів
    palette = [CLUSTER_PALETTE[index] / 255.0 for index in range(k)]
    cmap = ListedColormap(palette)
    figure, axis = plt.subplots(figsize=figure_size(rgb.shape, max_width=8.5))
    image = axis.imshow(labels, cmap=cmap, vmin=-0.5, vmax=k - 0.5, interpolation="nearest")
    axis.set_title("Карта кластерів, k-середні в просторі a*b*")
    axis.axis("off")
    colorbar = figure.colorbar(image, ax=axis, ticks=range(k), shrink=0.85)
    colorbar.ax.set_yticklabels([f"Кластер {index + 1}" for index in range(k)])
    save_figure(figure, OUT_DIR / "segmentation_labels.png")

    # розсіювання в просторі a*b*
    rng = np.random.default_rng(SEED)
    flat_ab = result["lab"][:, :, 1:3].reshape(-1, 2)
    flat_labels = labels.ravel()
    sample_size = min(30000, flat_ab.shape[0])
    sample_index = rng.choice(flat_ab.shape[0], size=sample_size, replace=False)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].scatter(
        flat_ab[sample_index, 0],
        flat_ab[sample_index, 1],
        s=3,
        c="0.55",
        alpha=0.35,
        linewidths=0,
    )
    axes[0].scatter(
        centers[:, 0],
        centers[:, 1],
        s=120,
        marker="*",
        c="tab:red",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        label="Центри кластерів",
    )
    axes[0].set_title("Простір a*b* до кластеризації")
    axes[0].set_xlabel("a* (зелений - червоний)")
    axes[0].set_ylabel("b* (синій - жовтий)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    for index in range(k):
        mask = flat_labels[sample_index] == index
        axes[1].scatter(
            flat_ab[sample_index][mask, 0],
            flat_ab[sample_index][mask, 1],
            s=3,
            color=palette[index],
            alpha=0.55,
            linewidths=0,
            label=f"Кластер {index + 1}",
        )
    axes[1].scatter(
        centers[:, 0],
        centers[:, 1],
        s=140,
        marker="*",
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    axes[1].set_title(f"Простір a*b* після k-середніх, k={k}")
    axes[1].set_xlabel("a*")
    axes[1].set_ylabel("b*")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, markerscale=2.5)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "segmentation_lab_scatter.png")

    # окремо кожен кластер і спільна панель
    figure, axes = plt.subplots(1, k, figsize=(4.2 * k, 4.4))
    if k == 1:
        axes = [axes]
    for index in range(k):
        segmented = rgb.copy()
        segmented[labels != index] = 0
        save_rgb(OUT_DIR / f"segmentation_cluster_{index + 1}.png", segmented)
        info = clusters[index]
        image_axis(
            axes[index],
            segmented,
            f"Кластер {index + 1}: a*={info['center_a']:.1f}, b*={info['center_b']:.1f}, "
            f"{100.0 * info['pixel_share']:.1f}%",
        )
    figure.suptitle("Сегментовані зображення за кластерами", fontsize=12)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "segmentation_clusters.png")

    # маска і контури вибраного кластера
    mask_raw = (labels == selected).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    selected_mask = mask_clean > 0
    overlay[selected_mask] = rgb[selected_mask]
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    image_axis(axes[0], rgb, "Вхідне зображення")
    image_axis(axes[1], mask_clean, f"Маска кластера {selected + 1}", cmap="gray")
    image_axis(
        axes[2],
        overlay,
        f"Виділений кластер {selected + 1}, найбільша насиченість",
    )
    figure.suptitle("Виділення вибраного кластера", fontsize=12)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "segmentation_selected_cluster.png")

    print(f"\nкластери k-середніх у просторі a*b*, k={k}:")
    print(
        f"{'кл.':>3} {'a*':>8} {'b*':>8} {'chroma':>8} {'pixels':>9} {'share':>7} "
        f"{'mean R':>8} {'mean G':>8} {'mean B':>8}"
    )
    for info in clusters:
        mean_rgb = info["mean_rgb"]
        print(
            f"{info['cluster']:>3} {info['center_a']:>8.2f} {info['center_b']:>8.2f} "
            f"{info['chroma']:>8.2f} {info['pixel_count']:>9d} "
            f"{100.0 * info['pixel_share']:>6.2f}% "
            f"{mean_rgb[0]:>8.1f} {mean_rgb[1]:>8.1f} {mean_rgb[2]:>8.1f}"
        )
    print(
        f"всього пікселів: {int(counts.sum())} із {total}, "
        f"компактність: {result['compactness']:.1f}"
    )
    print(f"вибраний кластер: {selected + 1}, найбільша насиченість")

    return {
        "k": k,
        "compactness": float(result["compactness"]),
        "selected": selected + 1,
        "clusters": clusters,
    }


def save_summary_figure(circle_info, seg_info):
    # підсумковий рисунок з основними цифрами
    radii = np.array([row["radius"] for row in circle_info["rows"]], dtype=np.float64)
    supports = np.array([row["edge_support"] for row in circle_info["rows"]], dtype=np.float64)
    shares = np.array([info["pixel_share"] for info in seg_info["clusters"]], dtype=np.float64)
    labels = [f"Кластер {info['cluster']}" for info in seg_info["clusters"]]

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))

    axes[0, 0].hist(radii, bins=10, color="tab:blue", edgecolor="white")
    axes[0, 0].set_title(f"Радіуси кіл, n={len(radii)}")
    axes[0, 0].set_xlabel("Радіус, пікселі")
    axes[0, 0].set_ylabel("Кількість")
    axes[0, 0].grid(alpha=0.25)

    sorted_radii = np.sort(radii)
    axes[0, 1].bar(np.arange(1, len(sorted_radii) + 1), sorted_radii, color="tab:green")
    axes[0, 1].set_title("Відсортовані радіуси кіл")
    axes[0, 1].set_xlabel("Номер кола")
    axes[0, 1].set_ylabel("Радіус, пікселі")
    axes[0, 1].grid(alpha=0.25, axis="y")

    axes[1, 0].barh(
        labels,
        shares * 100.0,
        color=[CLUSTER_PALETTE[i] / 255.0 for i in range(len(shares))],
    )
    axes[1, 0].set_title("Частки кластерів у зображенні")
    axes[1, 0].set_xlabel("Частка пікселів, %")
    axes[1, 0].grid(alpha=0.25, axis="x")

    axes[1, 1].hist(supports, bins=8, color="tab:orange", edgecolor="white")
    axes[1, 1].set_title("Підтримка контуру знайдених кіл")
    axes[1, 1].set_xlabel("Нормована підтримка градієнтами")
    axes[1, 1].set_ylabel("Кількість")
    axes[1, 1].grid(alpha=0.25)

    figure.suptitle("Підсумкові метрики", fontsize=13)
    figure.tight_layout()
    save_figure(figure, OUT_DIR / "summary_metrics.png")


def main():
    setup_console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("практична робота 8. пошук кіл і сегментація зображення")
    print(f"вхідні файли: {relative_path(DATA_DIR / 'circles.png')}, "
          f"{relative_path(DATA_DIR / 'segmentation.png')}")

    circles_path = DATA_DIR / "circles.png"
    segmentation_path = DATA_DIR / "segmentation.png"
    rgb_circles = load_rgb(circles_path)
    rgb_segmentation = load_rgb(segmentation_path)
    print(f"circles.png: {rgb_circles.shape[1]}x{rgb_circles.shape[0]} пікселів")
    print(f"segmentation.png: {rgb_segmentation.shape[1]}x{rgb_segmentation.shape[0]} пікселів")

    print("\nпараметри детектора кіл:")
    print(f"  dp = {HOUGH_PARAMS['dp']}")
    print(f"  minDist = {HOUGH_PARAMS['minDist']}")
    print(f"  param1 = {HOUGH_PARAMS['param1']}")
    print(f"  param2 = {HOUGH_PARAMS['param2']}")
    print(f"  minRadius = {HOUGH_PARAMS['minRadius']}, maxRadius = {HOUGH_PARAMS['maxRadius']}")
    print(
        f"  згладжування: gaussian {HOUGH_PARAMS['blur_kernel']}x{HOUGH_PARAMS['blur_kernel']}, "
        f"sigma={HOUGH_PARAMS['blur_sigma']}"
    )

    circle_info = run_circle_analysis(rgb_circles)
    seg_info = run_segmentation(rgb_segmentation)
    save_summary_figure(circle_info, seg_info)

    figures = {
        "circles_original": OUT_DIR / "circles_original.png",
        "circles_gray": OUT_DIR / "circles_gray.png",
        "circles_preprocessed": OUT_DIR / "circles_preprocessed.png",
        "circles_annotated": OUT_DIR / "circles_annotated.png",
        "circles_hough_overlay": OUT_DIR / "circles_hough_overlay.png",
        "circles_center_radius": OUT_DIR / "circles_center_radius.png",
        "circles_sensitivity": OUT_DIR / "circles_sensitivity.png",
        "segmentation_original": OUT_DIR / "segmentation_original.png",
        "segmentation_lab_scatter": OUT_DIR / "segmentation_lab_scatter.png",
        "segmentation_labels": OUT_DIR / "segmentation_labels.png",
        "segmentation_clusters": OUT_DIR / "segmentation_clusters.png",
        "segmentation_selected_cluster": OUT_DIR / "segmentation_selected_cluster.png",
        "summary_metrics": OUT_DIR / "summary_metrics.png",
    }
    for index in range(int(KMEANS_PARAMS["k"])):
        name = f"segmentation_cluster_{index + 1}.png"
        figures[f"segmentation_cluster_{index + 1}"] = OUT_DIR / name

    stats = circle_info["stats"]
    results = {
        "task": "praktychna-08",
        "title": "Пошук кіл і сегментація кольорового зображення",
        "inputs": {
            "circles": relative_path(circles_path),
            "segmentation": relative_path(segmentation_path),
        },
        "parameters": {
            "hough": dict(HOUGH_PARAMS),
            "kmeans": {
                "k": int(KMEANS_PARAMS["k"]),
                "attempts": int(KMEANS_PARAMS["attempts"]),
                "max_iter": int(KMEANS_PARAMS["max_iter"]),
                "epsilon": float(KMEANS_PARAMS["epsilon"]),
                "seed": SEED,
            },
        },
        "circles": {
            "count": int(circle_info["count"]),
            "radius_min": stats["min"],
            "radius_max": stats["max"],
            "radius_mean": stats["mean"],
            "radius_std": stats["std"],
            "radius_median": stats["median"],
        },
        "clusters": {
            "k": int(seg_info["k"]),
            "compactness": seg_info["compactness"],
            "selected": int(seg_info["selected"]),
            "items": [
                {
                    "cluster": int(info["cluster"]),
                    "center_a": info["center_a"],
                    "center_b": info["center_b"],
                    "pixel_count": int(info["pixel_count"]),
                    "pixel_share": info["pixel_share"],
                }
                for info in seg_info["clusters"]
            ],
        },
        "outputs": {
            "figures": {name: relative_path(path) for name, path in figures.items()},
            "tables": {"circles_csv": relative_path(OUT_DIR / "circles.csv")},
            "results_json": relative_path(OUT_DIR / "results.json"),
        },
    }

    with (OUT_DIR / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(f"\nрезультати збережено: {relative_path(OUT_DIR / 'results.json')}")
    print("готово")


if __name__ == "__main__":
    main()
