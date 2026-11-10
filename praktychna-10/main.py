# практична робота 10. розпізнавання облич засобами python
# детектор облич yunet і розпізнавач sface з opencv

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from faces import (
    align_and_feature,
    create_detector,
    create_recognizer,
    detect_faces,
    draw_face_box,
    imread_bgr,
    imwrite_bgr,
    match_features,
)
from utils import image_axis, save_figure

# utils перемикає matplotlib на agg, тому pyplot імпортуємо після нього
import matplotlib.pyplot as plt

WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR / "data"
OUT_DIR = WORK_DIR / "out"
REFERENCE_DIR = DATA_DIR / "reference"
DEFAULT_IMAGE = DATA_DIR / "test_group.jpg"

# еталони, порядок фіксований
REFERENCE_FILES = [
    ("Obama", "obama.jpg"),
    ("Biden", "biden.jpg"),
]

# поріг косинусної подібності для sface
THRESHOLD = 0.363
UNKNOWN_LABEL = "Unknown"
WEBCAM_FRAMES = 120


def rel_path(path):
    # шлях відносно каталогу практичної роботи
    path = Path(path).resolve()
    try:
        return path.relative_to(WORK_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def save_json(path, data):
    # зберігаємо json у utf-8
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_file(value):
    # шукаємо файл спершу від поточного каталогу, потім від каталогу роботи
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"файл не знайдено: {path}")

    for base in (Path.cwd(), WORK_DIR, DATA_DIR):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"файл не знайдено: {value}")


def biggest_face(faces):
    # з кількох облич беремо найбільше
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def load_references(detector, recognizer):
    # читаємо еталони і рахуємо для них вектори ознак
    references = []
    for name, filename in REFERENCE_FILES:
        path = REFERENCE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"немає еталону: {path}")

        image = imread_bgr(path)
        faces = detect_faces(detector, image)
        if len(faces) == 0:
            raise RuntimeError(f"на еталоні не знайдено облич: {path}")

        face = biggest_face(faces)
        embedding = align_and_feature(recognizer, image, face)

        # вирівняне обличчя зберігаємо окремо
        crop = OUT_DIR / f"reference_{name.lower()}.png"
        imwrite_bgr(crop, recognizer.alignCrop(image, face))

        references.append({
            "name": name,
            "path": path,
            "embedding": embedding.astype(np.float32),
            "crop": crop,
        })
        print(f"еталон {name}: {rel_path(path)}, вектор {embedding.shape}")

    return references


def recognize_faces(image, detector, recognizer, references):
    # шукаємо обличчя і порівнюємо їх з еталонами
    faces = detect_faces(detector, image)
    if len(faces) == 0:
        return []

    # сортуємо зліва направо, щоб результат був передбачуваним
    faces = faces[np.argsort(faces[:, 0])]

    matches = []
    for face in faces:
        embedding = align_and_feature(recognizer, image, face)
        similarities = {}
        for reference in references:
            similarities[reference["name"]] = match_features(
                recognizer, embedding, reference["embedding"]
            )

        best_name = max(similarities, key=similarities.get)
        best_similarity = float(similarities[best_name])
        recognized = best_similarity >= THRESHOLD

        matches.append({
            "face": face,
            "similarities": similarities,
            "best_name": best_name,
            "best_similarity": best_similarity,
            "label": best_name if recognized else UNKNOWN_LABEL,
            "recognized": recognized,
        })

    return matches


def annotate_image(image, matches):
    # малюємо рамки і підписи
    annotated = image.copy()
    for match in matches:
        color = (60, 180, 75) if match["recognized"] else (0, 165, 255)
        label = f"{match['label']} {match['best_similarity']:.3f}"
        draw_face_box(annotated, match["face"], label, color)
    return annotated


def face_info(matches, frame_index):
    # перетворюємо результати кадру на записи для json
    info = []
    for index, match in enumerate(matches):
        x, y, width, height = (int(round(value)) for value in match["face"][:4])
        info.append({
            "frame": frame_index,
            "index": index,
            "box": {"x": x, "y": y, "width": width, "height": height},
            "similarities": match["similarities"],
            "best_match": match["best_name"],
            "best_similarity": match["best_similarity"],
            "label": match["label"],
            "recognized": match["recognized"],
        })
    return info


def summarize(info):
    # короткий підсумок по обличчях
    names = []
    for face in info:
        if face["recognized"] and face["label"] not in names:
            names.append(face["label"])

    recognized = sum(1 for face in info if face["recognized"])
    return {
        "face_count": len(info),
        "recognized_count": recognized,
        "unknown_count": len(info) - recognized,
        "recognized_names": names,
    }


def save_gallery(references):
    # галерея вирівняних еталонних облич
    path = OUT_DIR / "reference_gallery.png"
    figure, axes = plt.subplots(
        1, len(references), figsize=(4.0 * len(references), 4.2)
    )
    for axis, reference in zip(np.atleast_1d(axes), references):
        crop = imread_bgr(reference["crop"])
        image_axis(axis, crop[:, :, ::-1], f"Еталон: {reference['name']}")
    figure.suptitle("Еталонні обличчя, вирівняні детектором YuNet", fontsize=12)
    save_figure(figure, path)
    print(f"галерея еталонів: {rel_path(path)}")
    return path


def save_similarity_matrix(references, info):
    # теплова карта подібностей облич до еталонів
    rows = [face for face in info if face["frame"] == 0][:12]
    if not rows:
        return None

    names = [reference["name"] for reference in references]
    matrix = np.array(
        [[face["similarities"][name] for name in names] for face in rows],
        dtype=np.float64,
    )

    path = OUT_DIR / "similarity_matrix.png"
    figure, axis = plt.subplots(
        figsize=(2.0 + 1.2 * len(names), 1.4 + 0.8 * len(rows))
    )
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, fontsize=10)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([f"Обличчя {index + 1}" for index in range(len(rows))])
    axis.set_title(
        f"Косинусна подібність до еталонів (поріг {THRESHOLD:.3f})",
        fontsize=11,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if value < 0.6 else "black"
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Подібність")
    save_figure(figure, path)
    print(f"матриця подібності: {rel_path(path)}")
    return path


def even_frame(frame):
    # mp4 просить парні розміри кадру
    height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


def open_writer(path, fps, size):
    # відкриваємо запис відео
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"не вдалося відкрити запис відео: {path}")
    return writer


def process_image(image_path, detector, recognizer, references):
    # обробляємо одне зображення
    image = imread_bgr(image_path)
    matches = recognize_faces(image, detector, recognizer, references)
    annotated = annotate_image(image, matches)

    stem = image_path.stem
    if stem.startswith("test_"):
        stem = stem[len("test_"):]
    output = OUT_DIR / f"recognized_{stem}.png"
    imwrite_bgr(output, annotated)

    info = face_info(matches, 0)

    print(f"джерело: {rel_path(image_path)}")
    print(f"знайдено облич: {len(info)}")
    for index, face in enumerate(info, start=1):
        details = ", ".join(
            f"{name}={value:.3f}" for name, value in face["similarities"].items()
        )
        print(
            f"  обличчя {index}: {details} -> {face['label']} "
            f"({face['best_similarity']:.3f})"
        )
    print(f"збережено: {rel_path(output)}")

    source = {
        "mode": "image",
        "path": rel_path(image_path),
        "frame_count": 1,
        "face_count": len(info),
    }
    return source, info, output


def process_video(
    capture,
    first_frame,
    video_path,
    mode,
    output_name,
    detector,
    recognizer,
    references,
    max_frames,
):
    # обробляємо кадри відео і зберігаємо анотоване відео
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1.0:
        fps = 20.0
    fps = min(max(fps, 5.0), 60.0)

    size = even_frame(first_frame).shape[1::-1]
    output = OUT_DIR / output_name
    writer = open_writer(output, fps, size)

    info = []
    frame_index = 0
    frame = first_frame
    try:
        while frame is not None:
            matches = recognize_faces(frame, detector, recognizer, references)
            annotated = annotate_image(frame, matches)
            writer.write(even_frame(annotated))
            info.extend(face_info(matches, frame_index))
            frame_index += 1
            if max_frames and frame_index >= max_frames:
                break
            success, frame = capture.read()
            if not success:
                frame = None
    finally:
        writer.release()

    if frame_index == 0:
        raise RuntimeError("не вдалося прочитати жодного кадру відео")

    source = {
        "mode": mode,
        "path": rel_path(video_path) if video_path is not None else None,
        "frame_count": frame_index,
        "face_count": len(info),
    }
    print(f"джерело: {mode}, кадрів оброблено: {frame_index}")
    print(f"знайдено облич: {len(info)}")
    print(f"збережено: {rel_path(output)}")
    return source, info, output


def run(args):
    # повний сценарій роботи
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("практична робота 10. розпізнавання облич (yunet + sface)")
    print(f"поріг подібності: {THRESHOLD:.3f}")

    detector = create_detector(640, 480)
    recognizer = create_recognizer()
    references = load_references(detector, recognizer)
    gallery = save_gallery(references)

    image_output = None
    video_output = None

    if args.image:
        source, info, image_output = process_image(
            find_file(args.image), detector, recognizer, references
        )
    elif args.source:
        video_path = find_file(args.source)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"не вдалося відкрити відеофайл: {video_path}")
        try:
            success, first_frame = capture.read()
            if not success or first_frame is None:
                raise RuntimeError(f"у відеофайлі немає кадрів: {video_path}")
            source, info, video_output = process_video(
                capture,
                first_frame,
                video_path,
                "video",
                f"annotated_{video_path.stem}.mp4",
                detector,
                recognizer,
                references,
                args.max_frames,
            )
        finally:
            capture.release()
    else:
        # типовий режим: вебкамера
        capture = cv2.VideoCapture(0)
        first_frame = None
        if capture.isOpened():
            success, first_frame = capture.read()
            if not success:
                first_frame = None

        if first_frame is None:
            capture.release()
            print("камеру не відкрито, обробляємо тестове зображення")
            source, info, image_output = process_image(
                DEFAULT_IMAGE, detector, recognizer, references
            )
        else:
            try:
                source, info, video_output = process_video(
                    capture,
                    first_frame,
                    None,
                    "webcam",
                    "annotated_webcam.mp4",
                    detector,
                    recognizer,
                    references,
                    args.max_frames or WEBCAM_FRAMES,
                )
            finally:
                capture.release()

    summary = summarize(info)
    matrix = save_similarity_matrix(references, info)

    results_path = OUT_DIR / "results.json"
    outputs = {
        "annotated_image": rel_path(image_output) if image_output else None,
        "annotated_video": rel_path(video_output) if video_output else None,
        "reference_crops": [rel_path(reference["crop"]) for reference in references],
        "reference_gallery": rel_path(gallery),
        "similarity_matrix": rel_path(matrix) if matrix else None,
        "results_json": rel_path(results_path),
    }

    result = {
        "parameters": {
            "detector": "YuNet (face_detection_yunet_2023mar.onnx)",
            "recognizer": "SFace (face_recognition_sface_2021dec.onnx)",
            "similarity": "cosine",
            "threshold": THRESHOLD,
        },
        "source": source,
        "references": [reference["name"] for reference in references],
        "faces": info,
        "summary": summary,
        "outputs": outputs,
    }
    save_json(results_path, result)

    print(
        f"підсумок: облич {summary['face_count']}, "
        f"розпізнано {summary['recognized_count']}, "
        f"невідомих {summary['unknown_count']}"
    )
    if summary["recognized_names"]:
        print("розпізнані особи: " + ", ".join(summary["recognized_names"]))
    print(f"результати: {rel_path(results_path)}")
    return result


def main():
    # запуск: python main.py --image data/test_group.jpg
    # щоб український текст не ламався у консолі
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="розпізнавання облич yunet + sface, без камери береться тестове зображення"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--image",
        metavar="PATH",
        help="обробити одне зображення, наприклад data/test_group.jpg",
    )
    group.add_argument(
        "--source",
        metavar="PATH",
        help="обробити відеофайл",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="скільки кадрів обробити, 0 - усі кадри",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames не може бути від'ємним")

    try:
        run(args)
    except Exception as error:
        print(f"помилка: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
