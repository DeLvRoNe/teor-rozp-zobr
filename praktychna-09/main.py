# практична робота 9
# розпізнавання обличчя і відстеження руху алгоритмом KLT
# запуск: python main.py --source data/test_face_video.avi --max-frames 60

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

from faces import YUNET_MODEL, create_detector, detect_faces, imwrite_bgr
from utils import save_figure

import matplotlib.pyplot as plt

WORK_DIR = Path(__file__).resolve().parent
DATA_DIR = WORK_DIR / "data"
OUT_DIR = WORK_DIR / "out"

# детектор yunet
DETECTOR_SCORE_THRESHOLD = 0.85
DETECTOR_NMS_THRESHOLD = 0.3
DETECTOR_TOP_K = 5000

# точки ши-томасі
MAX_CORNERS = 100
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
BLOCK_SIZE = 7
FALLBACK_QUALITY_LEVEL = 0.005
FALLBACK_MIN_DISTANCE = 5

# алгоритм лукаса-канаде
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_MAX_ITERATIONS = 30
LK_EPSILON = 0.01
FB_MAX_ERROR_PX = 2.0

# оцінювання перетворення
RANSAC_MAX_DISTANCE_PX = 4.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995
RANSAC_REFINE_ITERS = 10

# скільки точок треба для руху
MIN_POINTS_FOR_TRANSFORM = 2
MIN_POINTS_REDETECT = 6
DETERMINISTIC_SEED = 12345

WEBCAM_DETECTION_ATTEMPTS = 300
DEFAULT_MIDDLE_FRAME = 30
PROGRESS_STEP = 10

BOX_COLOR = (0, 255, 0)
LOST_COLOR = (0, 165, 255)
POINT_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)


def configure_console_encoding():
    # щоб український текст не ламався у консолі
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def rel_path(path):
    # шлях відносно папки практичної роботи
    path = Path(path).resolve()
    try:
        return path.relative_to(WORK_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def to_json(value):
    # перетворюємо numpy числа у звичайні для json
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def rnd(value, digits=4):
    # округлюємо число для json
    return round(float(value), digits)


def build_parser():
    # параметри командного рядка
    parser = argparse.ArgumentParser(
        description="Виявлення обличчя детектором YuNet і відстеження руху алгоритмом KLT"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="шлях до вхідного відео, без нього береться вебкамера",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        dest="max_frames",
        help="скільки кадрів обробляти (0 - всі)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="куди зберегти відео, типово out/tracked_face.avi",
    )
    return parser


def resolve_source(source):
    # шукаємо вхідний файл, для вебкамери повертаємо None
    if source is None:
        return None

    path = Path(source).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            WORK_DIR / path,
            DATA_DIR / path,
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "; ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"не знайдено відеофайл '{source}', перевірено: {tried}")


def resolve_output(output):
    # визначаємо куди пишемо відео
    if output is None:
        path = OUT_DIR / "tracked_face.avi"
    else:
        candidate = Path(output).expanduser()
        path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )

    if path.suffix.lower() != ".avi":
        raise ValueError("вихідне відео має бути з розширенням .avi")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_capture(source_path):
    # відкриваємо відеофайл або вебкамеру
    if source_path is None:
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("не вдалося відкрити вебкамеру VideoCapture(0)")
        return capture

    if not source_path.is_file():
        raise FileNotFoundError(f"не знайдено відеофайл: {source_path}")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"не вдалося відкрити відеофайл: {source_path}")
    return capture


def video_properties(capture, fallback_frame=None):
    # ширина, висота, fps і скільки кадрів у відео
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    if fallback_frame is not None:
        if width <= 0:
            width = int(fallback_frame.shape[1])
        if height <= 0:
            height = int(fallback_frame.shape[0])

    if width <= 0 or height <= 0:
        raise RuntimeError("не вдалося визначити розмір кадру")
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0
    if reported < 0:
        reported = 0
    return width, height, fps, reported


def detect_largest_face(detector, frame):
    # шукаємо обличя і беремо найбільше за площею
    faces = detect_faces(detector, frame)
    if len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def acquire_initial_face(capture, frame, detector, is_webcam):
    # чекаємо на обличчя, для вебкамери читаємо кадри далі
    source_index = 0
    attempt = 0
    while True:
        face = detect_largest_face(detector, frame)
        if face is not None:
            return frame, face, detector, source_index

        if not is_webcam:
            raise RuntimeError("не виявлено обличчя у першому кадрі відео")

        attempt += 1
        if attempt > WEBCAM_DETECTION_ATTEMPTS:
            raise RuntimeError("вебкамера не знайшла обличчя")

        success, next_frame = capture.read()
        if not success or next_frame is None:
            raise RuntimeError("вебкамера закінчила кадри до появи обличчя")

        size_changed = (
            next_frame.shape[1] != frame.shape[1]
            or next_frame.shape[0] != frame.shape[0]
        )
        if size_changed:
            height, width = next_frame.shape[:2]
            detector = create_detector(
                width,
                height,
                DETECTOR_SCORE_THRESHOLD,
                DETECTOR_NMS_THRESHOLD,
                DETECTOR_TOP_K,
            )
        frame = next_frame
        source_index += 1


def polygon_from_face(face):
    # прямокутник обличчя перетворюємо у чотирикутник
    x, y, width, height = (float(value) for value in face[:4])
    return np.array(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float32,
    )


def polygon_to_rect(polygon):
    # обмежувальний прямокутник чотирикутника
    return cv2.boundingRect(np.round(polygon).astype(np.int32))


def detect_trackable_features(gray, polygon):
    # шукаємо кути ши-томасі всередині обличчя
    height, width = gray.shape[:2]
    x, y, box_width, box_height = polygon_to_rect(polygon)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + max(box_width, 1))
    y1 = min(height, y + max(box_height, 1))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 1, 2), dtype=np.float32)

    mask = np.zeros_like(gray)
    mask[y0:y1, x0:x1] = 255

    def attempt(quality, distance):
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=quality,
            minDistance=distance,
            blockSize=BLOCK_SIZE,
            mask=mask,
            useHarrisDetector=False,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

    points = attempt(QUALITY_LEVEL, MIN_DISTANCE)
    if len(points) < MIN_POINTS_REDETECT:
        fallback = attempt(FALLBACK_QUALITY_LEVEL, FALLBACK_MIN_DISTANCE)
        if len(fallback) > len(points):
            points = fallback

    if len(points) < MIN_POINTS_FOR_TRANSFORM:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points


def track_points(previous_gray, gray, points):
    # трекаємо точки прямим і зворотним потоком
    if len(points) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty, 0

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        LK_MAX_ITERATIONS,
        LK_EPSILON,
    )
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        points,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=criteria,
    )
    if next_points is None or status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty, int(len(points))

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        next_points,
        None,
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=criteria,
    )
    if back_points is None or back_status is None:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        return empty, empty, int(len(points))

    forward_ok = status.reshape(-1) == 1
    backward_ok = back_status.reshape(-1) == 1
    backward_error = np.linalg.norm(
        back_points.reshape(-1, 2) - points.reshape(-1, 2),
        axis=1,
    )
    keep = forward_ok & backward_ok & (backward_error <= FB_MAX_ERROR_PX)

    old_points = points.reshape(-1, 2)[keep].reshape(-1, 1, 2)
    new_points = next_points.reshape(-1, 2)[keep].reshape(-1, 1, 2)
    rejected = int(len(points) - len(old_points))
    return old_points.astype(np.float32), new_points.astype(np.float32), rejected


def estimate_similarity(old_points, new_points):
    # шукаємо перетворення подібності методом ransac
    if len(old_points) < MIN_POINTS_FOR_TRANSFORM:
        return None, None
    if len(new_points) < MIN_POINTS_FOR_TRANSFORM:
        return None, None

    matrix, inliers = cv2.estimateAffinePartial2D(
        old_points.reshape(-1, 1, 2),
        new_points.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_MAX_DISTANCE_PX,
        maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE,
        refineIters=RANSAC_REFINE_ITERS,
    )
    if matrix is None or inliers is None:
        return None, None

    mask = inliers.reshape(-1).astype(bool)
    if int(mask.sum()) < MIN_POINTS_FOR_TRANSFORM:
        return None, None
    return np.asarray(matrix, dtype=np.float64), mask


def transform_polygon(polygon, matrix):
    # рухаємо чотирикутник обличчя за матрицею
    transformed = cv2.transform(
        polygon.reshape(1, -1, 2).astype(np.float32),
        matrix,
    )
    return transformed.reshape(-1, 2).astype(np.float32)


def recover_tracking(gray, frame, polygon, detector):
    # коли рух не оцінили то знову шукаємо точки і обличчя
    points = detect_trackable_features(gray, polygon)
    if len(points) >= MIN_POINTS_FOR_TRANSFORM:
        return points, polygon, True, False

    face = detect_largest_face(detector, frame)
    if face is None:
        return None, polygon, True, False

    new_polygon = polygon_from_face(face)
    points = detect_trackable_features(gray, new_polygon)
    if len(points) >= MIN_POINTS_FOR_TRANSFORM:
        return points, new_polygon, True, True
    return None, new_polygon, True, True


def draw_tracking(frame, polygon, points, frame_number, success, status_text):
    # малюємо рамку обличчя і точки
    annotated = frame.copy()
    color = BOX_COLOR if success else LOST_COLOR
    cv2.polylines(
        annotated,
        [np.round(polygon).astype(np.int32)],
        isClosed=True,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    if points is not None and len(points) > 0:
        height, width = annotated.shape[:2]
        for x, y in points.reshape(-1, 2):
            point = (int(round(float(x))), int(round(float(y))))
            if 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.drawMarker(
                    annotated,
                    point,
                    POINT_COLOR,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=8,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )

    count = 0 if points is None else int(len(points))
    text = f"frame {frame_number} | points {count} | {status_text}"
    cv2.putText(
        annotated,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return annotated


class VideoWriter:
    # пишемо відео з розміченими кадрами

    def __init__(self, destination, fps, size):
        destination.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(
            str(destination),
            fourcc,
            float(fps),
            (int(size[0]), int(size[1])),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"не вдалося створити відео: {destination}")
        self.frames_written = 0

    def write(self, frame):
        self.writer.write(frame)
        self.frames_written += 1

    def close(self):
        self.writer.release()


def make_tracking_summary(details, path):
    # рисунок зі кількістю точок по кадрах
    frames = [record["frame"] for record in details]
    visible = [record["visible_points"] for record in details]
    inliers = [record["inliers"] for record in details]
    failed_frames = [
        record["frame"]
        for record in details
        if record["frame"] > 0 and not record["transform_success"]
    ]

    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    axis.plot(frames, visible, color="tab:blue", label="видимі точки klt")
    axis.plot(frames, inliers, color="tab:green", label="інлайри ransac")
    if failed_frames:
        axis.scatter(
            failed_frames,
            [0] * len(failed_frames),
            color="tab:red",
            marker="x",
            label="не вдалося оцінити рух",
        )
    axis.set_title("Стійкість відстеження обличчя алгоритмом KLT", fontsize=12)
    axis.set_xlabel("номер кадру")
    axis.set_ylabel("кількість точок")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    axis.set_ylim(bottom=0)
    save_figure(figure, path)


def tracking_statistics(details):
    # зведені показники по кадрах
    tracking = details[1:]
    attempted = len(tracking)
    successful = sum(1 for record in tracking if record["transform_success"])
    failed = attempted - successful
    visible = [record["visible_points"] for record in tracking]
    inliers = [
        record["inliers"] for record in tracking if record["transform_success"]
    ]

    if attempted == 0:
        success_rate = 0.0
    else:
        success_rate = successful / attempted

    return {
        "attempted_frames": int(attempted),
        "successful_frames": int(successful),
        "failed_frames": int(failed),
        "success_rate": rnd(success_rate),
        "min_visible_points": int(min(visible)) if visible else 0,
        "max_visible_points": int(max(visible)) if visible else 0,
        "mean_visible_points": rnd(np.mean(visible)) if visible else 0.0,
        "min_inliers": int(min(inliers)) if inliers else 0,
        "max_inliers": int(max(inliers)) if inliers else 0,
        "mean_inliers": rnd(np.mean(inliers)) if inliers else 0.0,
        "feature_redetections": int(
            sum(1 for record in details if record["feature_redetected"])
        ),
        "face_redetections": int(
            sum(1 for record in details if record["face_redetected"])
        ),
        "bidirectional_rejected_points": int(
            sum(record["fb_rejected_points"] for record in details)
        ),
        "frames_without_points": int(
            sum(1 for record in details if record["points_after"] == 0)
        ),
    }


def run(args):
    # виявлення обличчя і відстеження руху
    if args.max_frames < 0:
        raise ValueError("параметр --max-frames не може бути від'ємним")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = resolve_output(args.output)
    source_path = resolve_source(args.source)
    is_webcam = source_path is None

    initial_frame_path = OUT_DIR / "frame_initial.png"
    middle_frame_path = OUT_DIR / "frame_middle.png"
    final_frame_path = OUT_DIR / "frame_final.png"
    summary_path = OUT_DIR / "tracking_summary.png"
    results_path = OUT_DIR / "results.json"

    capture = open_capture(source_path)
    writer = None
    try:
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError("не вдалося прочитати перший кадр")

        width, height, fps, reported_frames = video_properties(capture, frame)
        detector = create_detector(
            width,
            height,
            DETECTOR_SCORE_THRESHOLD,
            DETECTOR_NMS_THRESHOLD,
            DETECTOR_TOP_K,
        )

        frame, face, detector, first_source_index = acquire_initial_face(
            capture,
            frame,
            detector,
            is_webcam,
        )
        if frame.shape[1] != width or frame.shape[0] != height:
            width, height = frame.shape[1], frame.shape[0]

        initial_polygon = polygon_from_face(face)
        initial_points = detect_trackable_features(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            initial_polygon,
        )
        if len(initial_points) < MIN_POINTS_FOR_TRANSFORM:
            raise RuntimeError("після виявлення обличчя немає достатньо точок")

        cv2.setRNGSeed(DETERMINISTIC_SEED)

        current_polygon = initial_polygon.copy()
        current_points = initial_points
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_webcam:
            middle_target = (
                args.max_frames // 2
                if args.max_frames > 0
                else DEFAULT_MIDDLE_FRAME
            )
        else:
            expected_total = reported_frames
            if args.max_frames > 0:
                expected_total = min(expected_total, args.max_frames)
            middle_target = (
                expected_total // 2
                if expected_total > 0
                else DEFAULT_MIDDLE_FRAME
            )

        initial_annotation = draw_tracking(
            frame,
            current_polygon,
            current_points,
            frame_number=0,
            success=True,
            status_text="detected",
        )
        imwrite_bgr(initial_frame_path, initial_annotation)

        source_label = "вебкамера 0" if is_webcam else rel_path(source_path)
        print("практична робота 9. виявлення і відстеження обличчя алгоритмом KLT")
        print(f"джерело: {source_label}")
        print(f"розмір кадру: {width}x{height}, fps: {fps:.1f}")
        if reported_frames > 0:
            print(f"кадрів у джерелі: {reported_frames}")
        print(
            "детектор: yunet (cv2.FaceDetectorYN) замість "
            "matlab vision.CascadeObjectDetector"
        )
        print(
            f"точки ши-томасі: maxCorners={MAX_CORNERS}, "
            f"qualityLevel={QUALITY_LEVEL}, minDistance={MIN_DISTANCE}, "
            f"blockSize={BLOCK_SIZE}"
        )
        print(
            f"klt: winSize={LK_WIN_SIZE}, maxLevel={LK_MAX_LEVEL}, "
            f"критерій=({LK_MAX_ITERATIONS}, {LK_EPSILON}), "
            f"зворотна помилка<={FB_MAX_ERROR_PX:.1f} px"
        )
        print(
            f"початкове обличчя: кадр {first_source_index}, "
            f"bbox=({face[0]:.1f}, {face[1]:.1f}, {face[2]:.1f}, "
            f"{face[3]:.1f}), оцінка={face[14]:.4f}"
        )
        print(f"початкових точок: {len(initial_points)}")
        print("обробляємо кадри...")

        writer = VideoWriter(output_video, fps, (width, height))
        writer.write(initial_annotation)
        if 0 == middle_target:
            imwrite_bgr(middle_frame_path, initial_annotation)

        details = [
            {
                "frame": 0,
                "visible_points": int(len(initial_points)),
                "inliers": 0,
                "points_after": int(len(initial_points)),
                "transform_success": False,
                "feature_redetected": False,
                "face_redetected": False,
                "fb_rejected_points": 0,
            }
        ]
        final_annotation = initial_annotation
        processed = 1

        while args.max_frames == 0 or processed < args.max_frames:
            success, frame = capture.read()
            if not success or frame is None:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            visible_count = 0
            inlier_count = 0
            fb_rejected = 0
            redetected = False
            face_redetected = False
            transform_success = False

            enough_points = (
                current_points is not None
                and len(current_points) >= MIN_POINTS_FOR_TRANSFORM
            )
            if enough_points:
                old_points, new_points, fb_rejected = track_points(
                    previous_gray,
                    gray,
                    current_points,
                )
                visible_count = int(len(new_points))

                if len(new_points) >= MIN_POINTS_FOR_TRANSFORM:
                    matrix, inlier_mask = estimate_similarity(
                        old_points,
                        new_points,
                    )
                else:
                    matrix, inlier_mask = None, None

                if matrix is not None and inlier_mask is not None:
                    current_polygon = transform_polygon(current_polygon, matrix)
                    current_points = new_points.reshape(-1, 2)[inlier_mask].reshape(
                        -1, 1, 2
                    )
                    inlier_count = int(inlier_mask.sum())
                    transform_success = True

                    if len(current_points) < MIN_POINTS_REDETECT:
                        replenished = detect_trackable_features(
                            gray,
                            current_polygon,
                        )
                        if len(replenished) > len(current_points):
                            current_points = replenished
                            redetected = True
                else:
                    current_points, current_polygon, redetected, face_redetected = (
                        recover_tracking(
                            gray,
                            frame,
                            current_polygon,
                            detector,
                        )
                    )
            else:
                current_points, current_polygon, redetected, face_redetected = (
                    recover_tracking(
                        gray,
                        frame,
                        current_polygon,
                        detector,
                    )
                )

            status_text = "tracked" if transform_success else "lost"
            final_annotation = draw_tracking(
                frame,
                current_polygon,
                current_points,
                frame_number=processed,
                success=transform_success,
                status_text=status_text,
            )
            writer.write(final_annotation)

            points_after = 0 if current_points is None else int(len(current_points))
            details.append(
                {
                    "frame": int(processed),
                    "visible_points": int(visible_count),
                    "inliers": int(inlier_count),
                    "points_after": points_after,
                    "transform_success": bool(transform_success),
                    "feature_redetected": bool(redetected),
                    "face_redetected": bool(face_redetected),
                    "fb_rejected_points": int(fb_rejected),
                }
            )

            if processed == middle_target:
                imwrite_bgr(middle_frame_path, final_annotation)

            if processed % PROGRESS_STEP == 0 or not transform_success:
                print(
                    f"  кадр {processed}: видимих точок {visible_count}, "
                    f"інлайрів {inlier_count}, "
                    f"{'успіх' if transform_success else 'втрата'}"
                )

            previous_gray = gray
            processed += 1

        imwrite_bgr(final_frame_path, final_annotation)
        if not middle_frame_path.is_file():
            imwrite_bgr(middle_frame_path, final_annotation)
    finally:
        capture.release()
        if writer is not None:
            writer.close()

    output_video_frames = writer.frames_written

    tracking = tracking_statistics(details)
    if is_webcam:
        expected_frames = args.max_frames if args.max_frames > 0 else 0
    else:
        expected_frames = reported_frames
        if args.max_frames > 0:
            expected_frames = min(expected_frames, args.max_frames)

    make_tracking_summary(details, summary_path)

    results = {
        "source": "webcam:0" if is_webcam else rel_path(source_path),
        "source_type": "webcam" if is_webcam else "video",
        "opencv_version": cv2.__version__,
        "parameters": {
            "detector": {
                "implementation": "cv2.FaceDetectorYN (YuNet 2023mar)",
                "model": rel_path(YUNET_MODEL),
                "score_threshold": DETECTOR_SCORE_THRESHOLD,
                "nms_threshold": DETECTOR_NMS_THRESHOLD,
                "top_k": DETECTOR_TOP_K,
            },
            "features": {
                "method": "Shi-Tomasi (cv2.goodFeaturesToTrack)",
                "max_corners": MAX_CORNERS,
                "quality_level": QUALITY_LEVEL,
                "min_distance": MIN_DISTANCE,
                "block_size": BLOCK_SIZE,
                "fallback_quality_level": FALLBACK_QUALITY_LEVEL,
                "fallback_min_distance": FALLBACK_MIN_DISTANCE,
            },
            "klt": {
                "method": "cv2.calcOpticalFlowPyrLK",
                "win_size": [LK_WIN_SIZE[0], LK_WIN_SIZE[1]],
                "max_level": LK_MAX_LEVEL,
                "max_iterations": LK_MAX_ITERATIONS,
                "epsilon": LK_EPSILON,
                "bidirectional_error_px": FB_MAX_ERROR_PX,
            },
            "transform": {
                "model": "similarity",
                "method": "RANSAC (cv2.estimateAffinePartial2D)",
                "max_distance_px": RANSAC_MAX_DISTANCE_PX,
                "max_iters": RANSAC_MAX_ITERS,
                "confidence": RANSAC_CONFIDENCE,
                "refine_iters": RANSAC_REFINE_ITERS,
            },
            "max_frames": args.max_frames if args.max_frames > 0 else None,
        },
        "video": {
            "width": int(width),
            "height": int(height),
            "fps": rnd(fps, 3),
            "reported_frame_count": int(reported_frames),
            "codec": "MJPG",
        },
        "frames": {
            "expected": int(expected_frames),
            "processed": int(processed),
            "successful_transforms": int(tracking["successful_frames"]),
            "failed_transforms": int(tracking["failed_frames"]),
            "output_video_frames": int(output_video_frames),
        },
        "detection": {
            "initial_frame_index": int(first_source_index),
            "initial_bbox": [rnd(value, 3) for value in face[:4]],
            "initial_score": rnd(face[14], 4),
            "initial_feature_count": int(len(initial_points)),
            "face_detections_total": int(1 + tracking["face_redetections"]),
        },
        "tracking": tracking,
        "outputs": {
            "video": rel_path(output_video),
            "initial_frame": rel_path(initial_frame_path),
            "middle_frame": rel_path(middle_frame_path),
            "final_frame": rel_path(final_frame_path),
            "summary_figure": rel_path(summary_path),
            "results": rel_path(results_path),
        },
    }

    results_path.write_text(
        json.dumps(to_json(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"оброблено кадрів: {processed}; успішних перетворень: "
        f"{tracking['successful_frames']} з {tracking['attempted_frames']} "
        f"({100.0 * tracking['success_rate']:.1f} %)"
    )
    print(
        f"середня кількість видимих точок: {tracking['mean_visible_points']:.1f}; "
        f"інлайрів: {tracking['mean_inliers']:.1f}"
    )
    print(f"відео: {rel_path(output_video)}")
    print(f"початковий кадр: {rel_path(initial_frame_path)}")
    print(f"середній кадр: {rel_path(middle_frame_path)}")
    print(f"кінцевий кадр: {rel_path(final_frame_path)}")
    print(f"рисунок: {rel_path(summary_path)}")
    print(f"результати: {rel_path(results_path)}")
    print("роботу завершено")
    return results


def main():
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("обробку перервано")
        return 130
    except Exception as error:
        print(f"помилка: {error}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
