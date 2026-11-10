# детектор облич yunet
# модель лежить у папці models, якщо її немає то качаємо з opencv zoo

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def model_path():
    # повертає шлях до моделі, завантажуємо її якщо немає
    if not YUNET_MODEL.is_file():
        print(f"качаємо модель yunet: {YUNET_URL}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, str(YUNET_MODEL))
    return YUNET_MODEL


def create_detector(width, height, score_threshold=0.9, nms_threshold=0.3, top_k=5000):
    # створюємо детектор yunet під розмір кадру
    return cv2.FaceDetectorYN.create(
        str(model_path()),
        "",
        (int(width), int(height)),
        float(score_threshold),
        float(nms_threshold),
        int(top_k),
    )


def detect_faces(detector, image):
    # повертає знайдені обличя масивом (N, 15)
    height, width = image.shape[:2]
    detector.setInputSize((int(width), int(height)))
    _, faces = detector.detect(image)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return np.asarray(faces, dtype=np.float32)


def imwrite_bgr(path, image):
    # зберігаємо зображення у форматі bgr
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix if path.suffix else ".png"
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"не вдалося зберегти зображення: {path}")
    buffer.tofile(str(path))
