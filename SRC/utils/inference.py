import gc
from statistics import median

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw

from .config import (
    CHAR_CONF,
    MAX_IMAGE_DIM,
    PLATE_CONF,
    OnnxYoloDetector,
    get_font,
    get_model,
    get_ocr_model,
)

_PLATE_STYLES: dict[str, tuple[tuple[int, int, int], str]] = {
    "car": ((16, 185, 129), " [Car]"),
    "motorcycle": ((245, 158, 11), " [Motorcycle]"),
    "unknown": ((16, 185, 129), ""),
}

ARABIC_DIGITS = set("٠١٢٣٤٥٦٧٨٩0123456789")


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR array and cap huge resolutions to save RAM."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes -- unsupported or corrupt image.")

    h, w = image.shape[:2]
    longest = max(h, w)
    if MAX_IMAGE_DIM > 0 and longest > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _ar(text: str) -> str:
    """Reshape and apply BiDi display ordering for Arabic PIL rendering."""
    return get_display(arabic_reshaper.reshape(text))


def _get_plate_class_map(names: dict[int, str]) -> dict[int, str]:
    """Map plate class IDs to vehicle types for both 28-class and 29-class models."""
    plate_map: dict[int, str] = {}
    for idx, raw_name in names.items():
        low = str(raw_name).strip().lower()
        if "plate" in low:
            if "moto" in low:
                plate_map[int(idx)] = "motorcycle"
            elif "car" in low:
                plate_map[int(idx)] = "car"
            else:
                plate_map[int(idx)] = "unknown"
    return plate_map or {0: "unknown"}


def _plate_style(vehicle_type: str) -> tuple[tuple[int, int, int], str]:
    """Return (outline RGB color, label tag) for a vehicle type."""
    return _PLATE_STYLES.get(vehicle_type, _PLATE_STYLES["unknown"])


def _yolo_onnx_predict(
    detector: OnnxYoloDetector,
    img_bgr: np.ndarray,
    conf_thres: float,
    iou_thres: float = 0.70,
) -> list[tuple[float, float, float, float, float, int]]:
    """Run Ultralytics-compatible letterbox + ONNX forward pass + class-aware NMS."""
    h0, w0 = img_bgr.shape[:2]
    imgsz = detector.imgsz

    r = min(imgsz / float(h0), imgsz / float(w0))
    new_w = max(1, int(round(w0 * r)))
    new_h = max(1, int(round(h0 * r)))
    dw = (imgsz - new_w) / 2.0
    dh = (imgsz - new_h) / 2.0

    resized = (
        cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (new_w, new_h) != (w0, h0)
        else img_bgr
    )
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb = padded[:, :, ::-1].transpose(2, 0, 1)
    dtype = np.float16 if detector.is_fp16 else np.float32
    blob = np.ascontiguousarray(rgb, dtype=dtype)
    blob /= dtype(255.0)
    blob = blob[None, ...]

    raw_out = detector.session.run(None, {detector.input_name: blob})[0]
    del blob, rgb, padded

    # Output shape: (1, 4 + nc, 8400) -> transpose to (8400, 4 + nc) in float32
    preds = raw_out[0].astype(np.float32, copy=False).T
    scores_matrix = preds[:, 4:]
    max_scores = scores_matrix.max(axis=1)
    mask = max_scores >= conf_thres
    if not np.any(mask):
        return []

    filtered = preds[mask]
    confs = max_scores[mask]
    cls_ids = scores_matrix[mask].argmax(axis=1)

    cx = filtered[:, 0]
    cy = filtered[:, 1]
    bw = filtered[:, 2]
    bh = filtered[:, 3]

    x1 = np.clip((cx - bw * 0.5 - left) / r, 0.0, float(w0))
    y1 = np.clip((cy - bh * 0.5 - top) / r, 0.0, float(h0))
    x2 = np.clip((cx + bw * 0.5 - left) / r, 0.0, float(w0))
    y2 = np.clip((cy + bh * 0.5 - top) / r, 0.0, float(h0))

    # Class-offset NMS (matches Ultralytics batched_nms with max_wh=7680)
    offsets = cls_ids.astype(np.float32) * 7680.0
    nms_boxes = [
        [float(x1[i] + offsets[i]), float(y1[i] + offsets[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])]
        for i in range(len(confs))
    ]
    kept_indices = cv2.dnn.NMSBoxes(
        nms_boxes, confs.tolist(), float(conf_thres), float(iou_thres)
    )
    if len(kept_indices) == 0:
        return []

    flat_idx = np.array(kept_indices, dtype=np.int32).reshape(-1)
    return [
        (
            float(x1[i]),
            float(y1[i]),
            float(x2[i]),
            float(y2[i]),
            float(confs[i]),
            int(cls_ids[i]),
        )
        for i in flat_idx
    ]


def _box_iou(a: dict, b: dict) -> float:
    """Compute IoU between two character box dicts."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1e-6, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / (area_a + area_b - inter)


def _dedup_chars(chars: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    """Greedy cross-class IoU suppression keeping highest-confidence character per slot."""
    kept: list[dict] = []
    for cand in sorted(chars, key=lambda c: c["conf"], reverse=True):
        if any(_box_iou(cand, prev) >= iou_thresh for prev in kept):
            continue
        kept.append(cand)
    return kept


def _extract_chars(
    ocr_model: OnnxYoloDetector,
    img_crop: np.ndarray,
    char_conf: float,
    ocr_plate_ids: set[int],
    offset_y: float = 0.0,
) -> list[dict]:
    """Run Stage-2 ONNX OCR on a crop or sub-band and filter tiny noise boxes."""
    if img_crop.size == 0:
        return []
    ch, cw = img_crop.shape[:2]
    boxes = _yolo_onnx_predict(ocr_model, img_crop, conf_thres=char_conf)

    extracted: list[dict] = []
    for x1, y1, x2, y2, conf, cls_id in boxes:
        if cls_id in ocr_plate_ids:
            continue
        bw, bh = x2 - x1, y2 - y1
        if bw < 0.02 * cw or bh < 0.04 * ch:
            continue
        extracted.append(
            {
                "x1": x1,
                "y1": y1 + offset_y,
                "x2": x2,
                "y2": y2 + offset_y,
                "x_mid": (x1 + x2) / 2.0,
                "y_mid": (y1 + y2) / 2.0 + offset_y,
                "h": bh,
                "conf": conf,
                "char": str(ocr_model.names[cls_id]),
            }
        )
    return extracted


def _sort_plate_chars(chars: list[dict], is_two_row: bool) -> list[dict]:
    """Sort characters right-to-left; for two-row plates put letters row first then digits row."""
    if len(chars) <= 1:
        return chars

    med_h = median(c["h"] for c in chars)
    y_mids = [c["y_mid"] for c in chars]
    y_span = max(y_mids) - min(y_mids)

    if (is_two_row or y_span > 0.65 * med_h) and y_span > 0.45 * med_h:
        by_y = sorted(chars, key=lambda c: c["y_mid"])
        gaps = [by_y[i + 1]["y_mid"] - by_y[i]["y_mid"] for i in range(len(by_y) - 1)]
        split_idx = max(range(len(gaps)), key=gaps.__getitem__)
        row_top = sorted(by_y[: split_idx + 1], key=lambda c: c["x_mid"], reverse=True)
        row_bot = sorted(by_y[split_idx + 1 :], key=lambda c: c["x_mid"], reverse=True)

        def _letter_ratio(row: list[dict]) -> float:
            return sum(1 for c in row if c["char"] not in ARABIC_DIGITS) / len(row)

        if _letter_ratio(row_bot) > _letter_ratio(row_top):
            return row_bot + row_top
        return row_top + row_bot

    return sorted(chars, key=lambda c: c["x_mid"], reverse=True)


def run_pipeline(
    image_bytes: bytes,
    plate_conf: float = PLATE_CONF,
    char_conf: float = CHAR_CONF,
) -> dict:
    """
    Run decoupled two-stage ONNX YOLO license plate + Arabic character recognition.

    Memory-optimized for 512 MB cloud containers (no PyTorch/Ultralytics dependency).
    """
    img_bgr = _decode_image(image_bytes)
    height, width = img_bgr.shape[:2]

    det_model = get_model()
    ocr_model = get_ocr_model()
    det_plate_map = _get_plate_class_map(det_model.names)
    ocr_plate_ids = set(_get_plate_class_map(ocr_model.names).keys())

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(img_pil.width)

    plate_boxes = _yolo_onnx_predict(det_model, img_bgr, conf_thres=plate_conf)
    raw_plates = [
        (
            (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            conf,
            det_plate_map[cls_id],
        )
        for x1, y1, x2, y2, conf, cls_id in plate_boxes
        if cls_id in det_plate_map
    ]

    used_fallback = False
    if not raw_plates:
        used_fallback = True
        raw_plates = [((0, 0, width, height), 1.0, "unknown")]

    detections: list[dict] = []
    plate_idx = 0

    for (x1, y1, x2, y2), conf, vehicle_type in raw_plates:
        pad_x = max(4, int((x2 - x1) * 0.06))
        pad_y = max(4, int((y2 - y1) * 0.06))
        crop = img_bgr[
            max(0, y1 - pad_y) : min(height, y2 + pad_y),
            max(0, x1 - pad_x) : min(width, x2 + pad_x),
        ]
        if crop.size == 0:
            continue

        ch, cw = crop.shape[:2]
        if min(cw, ch) < 96:
            scale = 128.0 / max(1, min(cw, ch))
            crop = cv2.resize(
                crop,
                (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
            ch, cw = crop.shape[:2]

        chars = _dedup_chars(_extract_chars(ocr_model, crop, char_conf, ocr_plate_ids))

        aspect = (x2 - x1) / max(1, y2 - y1)
        is_two_row = (vehicle_type == "motorcycle") or (aspect < 1.35)

        if is_two_row and len(chars) < 4:
            y0 = int(ch * 0.20)
            body_h = ch - y0
            if body_h >= 12:
                top_end = y0 + int(body_h * 0.58)
                bot_start = y0 + int(body_h * 0.42)
                band_top = _extract_chars(
                    ocr_model,
                    crop[y0:top_end, :],
                    char_conf,
                    ocr_plate_ids,
                    offset_y=float(y0),
                )
                band_bot = _extract_chars(
                    ocr_model,
                    crop[bot_start:, :],
                    char_conf,
                    ocr_plate_ids,
                    offset_y=float(bot_start),
                )
                band_chars = _dedup_chars(band_top + band_bot)
                if len(band_chars) > len(chars):
                    chars = band_chars

        if used_fallback and not chars:
            continue

        ordered_chars = _sort_plate_chars(chars, is_two_row=is_two_row)
        plate_idx += 1
        characters = [c["char"] for c in ordered_chars]
        text = " ".join(characters)
        precision = round(min(max(conf, 0.0), 1.0), 2)
        is_partial = len(characters) < 3

        outline_color, tag = _plate_style(vehicle_type)
        label = _ar(f"#{plate_idx}{tag} ({precision:.2f}) | {text}")
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        bw, bh = (r - l) + 20, (b - t) + 12
        bx = max(0, min(x1, max(0, img_pil.width - bw)))
        by = max(0, y1 - bh - 6)

        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=4)
        draw.rectangle(
            [bx, by, bx + bw, by + bh],
            fill=(15, 23, 42),
            outline=outline_color,
            width=2,
        )
        draw.text((bx + 10 - l, by + 6 - t), label, font=font, fill=(255, 255, 255))

        detections.append(
            {
                "plate_index": plate_idx,
                "vehicle_type": vehicle_type,
                "precision": precision,
                "text": text,
                "characters": characters,
                "bbox": [x1, y1, x2, y2],
                "fallback_full_image": used_fallback,
                "is_partial": is_partial,
            }
        )

    del img_bgr
    annotated_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    del img_pil
    success, encoded = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    del annotated_bgr
    gc.collect()

    if not success:
        raise ValueError("Failed to encode annotated image.")

    return {
        "plates": detections,
        "annotated_image": encoded.tobytes(),
    }
