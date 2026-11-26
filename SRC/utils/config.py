import os
import threading
import torch
from dotenv import load_dotenv
from PIL import ImageFont
from ultralytics import YOLO

load_dotenv()

SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(SRC_PATH, "artifacts")

APP_NAME = os.getenv("APP_NAME", "Egyptian License Plate Recognition API")
VERSION = os.getenv("VERSION", "1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(ARTIFACTS_DIR, "best.pt")
DEFAULT_OCR_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ocr_best.pt")
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH") or (
    DEFAULT_OCR_MODEL_PATH if os.path.isfile(DEFAULT_OCR_MODEL_PATH) else MODEL_PATH
)
DB_PATH = os.getenv("DB_PATH") or os.path.join(ARTIFACTS_DIR, "plates.db")
OUTPUT_DIR = os.getenv("OUTPUT_DIR") or os.path.join(ARTIFACTS_DIR, "outputs")

PLATE_CONF = float(os.getenv("PLATE_CONF") or 0.25)
CHAR_CONF = float(os.getenv("CHAR_CONF") or 0.10)
FONT_PATH = os.getenv("FONT_PATH", "").strip()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB") or 15)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

_model = None
_ocr_model = None
_model_lock = threading.Lock()


def is_model_loaded() -> bool:
    """Return whether the Stage-1 YOLO model singleton has been loaded."""
    return _model is not None


def get_device() -> int | str:
    """Resolve YOLO inference device from env or CUDA availability."""
    env_device = os.getenv("YOLO_DEVICE", "").strip()
    if env_device:
        return int(env_device) if env_device.isdigit() else env_device
    return 0 if torch.cuda.is_available() else "cpu"


def get_model() -> YOLO:
    """Return the thread-safe lazy-loaded Stage-1 plate detection YOLO singleton."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = YOLO(MODEL_PATH)
    return _model


def get_ocr_model() -> YOLO:
    """Return the Stage-2 character OCR YOLO singleton, reusing get_model() if identical."""
    if os.path.abspath(OCR_MODEL_PATH) == os.path.abspath(MODEL_PATH):
        return get_model()
    global _ocr_model
    if _ocr_model is None:
        with _model_lock:
            if _ocr_model is None:
                _ocr_model = YOLO(OCR_MODEL_PATH)
    return _ocr_model


def get_font(image_width: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load an Arabic-capable TrueType font scaled to image width."""
    size = max(18, image_width // 36)
    candidates = [
        *([] if not FONT_PATH else [FONT_PATH]),
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()
