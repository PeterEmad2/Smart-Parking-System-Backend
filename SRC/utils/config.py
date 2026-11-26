import ast
import json
import os
import threading

# Constrain thread pools & glibc arenas BEFORE importing C/C++ native extensions (critical for 512 MB Render containers)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import cv2
import onnxruntime as ort
from dotenv import load_dotenv
from PIL import ImageFont

cv2.setNumThreads(1)
load_dotenv()

SRC_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(SRC_PATH, "artifacts")

APP_NAME = os.getenv("APP_NAME", "Egyptian Smart Parking & ANPR API")
VERSION = os.getenv("VERSION", "2.0.0")


def _resolve_default_model(stem: str) -> str:
    onnx_candidate = os.path.join(ARTIFACTS_DIR, f"{stem}.onnx")
    if os.path.isfile(onnx_candidate):
        return onnx_candidate
    return os.path.join(ARTIFACTS_DIR, f"{stem}.pt")


MODEL_PATH = os.getenv("MODEL_PATH") or _resolve_default_model("best")
DEFAULT_OCR_PATH = _resolve_default_model("ocr_best")
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH") or (
    DEFAULT_OCR_PATH if os.path.isfile(DEFAULT_OCR_PATH) else MODEL_PATH
)

DB_PATH = os.getenv("DB_PATH") or os.path.join(ARTIFACTS_DIR, "plates.db")
OUTPUT_DIR = os.getenv("OUTPUT_DIR") or os.path.join(ARTIFACTS_DIR, "outputs")

PLATE_CONF = float(os.getenv("PLATE_CONF") or 0.25)
CHAR_CONF = float(os.getenv("CHAR_CONF") or 0.10)
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM") or 1600)
LOW_MEMORY_UNLOAD = os.getenv("LOW_MEMORY_UNLOAD", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
FONT_PATH = os.getenv("FONT_PATH", "").strip()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB") or 15)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)


class OnnxYoloDetector:
    """Lightweight ONNX Runtime wrapper for Ultralytics YOLO detection models."""

    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(os.getenv("ORT_THREADS", "2")))
        opts.inter_op_num_threads = 1
        # Release intermediate Conv2D buffers back to OS immediately after each run()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            and os.getenv("YOLO_DEVICE", "cpu").lower() != "cpu"
            else ["CPUExecutionProvider"]
        )

        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        in0 = self.session.get_inputs()[0]
        self.input_name: str = in0.name
        self.is_fp16: bool = "float16" in in0.type
        self.imgsz: int = (
            int(in0.shape[2])
            if len(in0.shape) >= 4 and isinstance(in0.shape[2], int)
            else 640
        )
        self.names: dict[int, str] = self._parse_names()

    def _parse_names(self) -> dict[int, str]:
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names", "{}")
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = json.loads(raw)
        return {int(k): str(v) for k, v in parsed.items()}


_model: OnnxYoloDetector | None = None
_ocr_model: OnnxYoloDetector | None = None
_model_lock = threading.Lock()


def is_model_loaded() -> bool:
    """Return whether Stage-1 model is loaded or present on disk ready for inference."""
    return _model is not None or os.path.isfile(MODEL_PATH)


def get_device() -> str:
    """Return active ONNX Runtime execution provider descriptor."""
    available = ort.get_available_providers()
    if (
        "CUDAExecutionProvider" in available
        and os.getenv("YOLO_DEVICE", "cpu").lower() != "cpu"
    ):
        return "onnxruntime:cuda"
    return "onnxruntime:cpu"


def get_model() -> OnnxYoloDetector:
    """Return the thread-safe lazy-loaded Stage-1 plate detection ONNX singleton."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = OnnxYoloDetector(MODEL_PATH)
    return _model


def get_ocr_model() -> OnnxYoloDetector:
    """Return the thread-safe lazy-loaded Stage-2 character OCR ONNX singleton."""
    if os.path.abspath(OCR_MODEL_PATH) == os.path.abspath(MODEL_PATH):
        return get_model()
    global _ocr_model
    if _ocr_model is None:
        with _model_lock:
            if _ocr_model is None:
                _ocr_model = OnnxYoloDetector(OCR_MODEL_PATH)
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
