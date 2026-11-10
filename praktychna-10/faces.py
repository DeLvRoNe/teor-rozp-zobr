# робота з моделями облич yunet і sface

import urllib.request
from pathlib import Path

import cv2
import numpy as np

# моделі лежать у каталозі models поруч із цим файлом
MODELS_DIR = Path(__file__).resolve().parent / "models"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# моделі беремо з opencv zoo
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def download_model(path, url):
    # качаємо модель, якщо її ще немає
    if path.is_file():
        return
    print(f"завантажуємо модель {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def imread_bgr(path):
    # читаємо зображення
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"не вдалося прочитати зображення: {path}")
    return image


def imwrite_bgr(path, image):
    # зберігаємо зображення
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix if path.suffix else ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"не вдалося зберегти зображення: {path}")
    buffer.tofile(str(path))


def create_detector(width, height, score_threshold=0.9, nms_threshold=0.3, top_k=5000):
    # детектор облич yunet
    download_model(YUNET_MODEL, YUNET_URL)
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL),
        "",
        (int(width), int(height)),
        float(score_threshold),
        float(nms_threshold),
        int(top_k),
    )


def create_recognizer():
    # розпізнавач облич sface
    download_model(SFACE_MODEL, SFACE_URL)
    return cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")


def detect_faces(detector, image):
    # шукаємо обличчя, повертаємо масив (n, 15)
    height, width = image.shape[:2]
    detector.setInputSize((int(width), int(height)))
    _, faces = detector.detect(image)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return np.asarray(faces, dtype=np.float32)


def align_and_feature(recognizer, image, face):
    # вирівнюємо обличчя і рахуємо вектор ознак
    aligned = recognizer.alignCrop(image, face)
    return recognizer.feature(aligned)


def match_features(recognizer, first, second):
    # косинусна подібність двох векторів ознак
    return float(
        recognizer.match(first, second, cv2.FaceRecognizerSF_FR_COSINE)
    )


def draw_face_box(image, face, label=None, color=(0, 255, 0)):
    # малюємо рамку обличчя і підпис
    x, y, width, height = (int(round(value)) for value in face[:4])
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    if label:
        text_y = max(y - 8, 16)
        cv2.putText(
            image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
