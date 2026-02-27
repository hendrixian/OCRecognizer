import os
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import (
    DATE_YOLO_WEIGHTS,
    DATE_DAY_MONTH_MODEL,
    DATE_YEAR_MODEL,
    DATE_CONF_THRESHOLD,
    DEFAULT_BURMESE_DIGITS,
    BURMESE_TO_LATIN
)


DEBUG_OCR = os.getenv("OCR_DEBUG", "0") == "1"


class NRCDateRecognizer:
    def __init__(
        self,
        detector_path: Path = DATE_YOLO_WEIGHTS,
        day_month_path: Path = DATE_DAY_MONTH_MODEL,
        year_path: Path = DATE_YEAR_MODEL,
        conf_threshold: float = DATE_CONF_THRESHOLD
    ):
        self.detector_path = Path(detector_path)
        self.day_month_path = Path(day_month_path)
        self.year_path = Path(year_path)
        self.conf_threshold = float(conf_threshold)

        self._ready = False
        self._load_error = None
        self._tf = None
        self._load_model = None
        self.detector = None
        self.day_month_model = None
        self.year_model = None
        self.day_month_labels = [self._to_burmese_number(value) for value in range(0, 32)]

    def _to_burmese_number(self, value):
        text = str(int(value))
        return ''.join(DEFAULT_BURMESE_DIGITS[int(digit)] for digit in text)

    def _ensure_loaded(self):
        if self._ready:
            return True
        if self._load_error:
            return False
        if not (self.detector_path.exists() and self.day_month_path.exists() and self.year_path.exists()):
            self._load_error = 'date models missing'
            if DEBUG_OCR:
                print(f"[ocr] date models missing: {self.detector_path}, {self.day_month_path}, {self.year_path}")
            return False

        try:
            import tensorflow as tf  # noqa: WPS433
            from tensorflow.keras.models import load_model  # noqa: WPS433
        except Exception as exc:
            self._load_error = f'tensorflow unavailable: {exc}'
            if DEBUG_OCR:
                print(f"[ocr] date recognizer disabled: {self._load_error}")
            return False

        try:
            self.detector = YOLO(str(self.detector_path))
            self.day_month_model = load_model(str(self.day_month_path))
            self.year_model = load_model(str(self.year_path))
        except Exception as exc:
            self._load_error = f'failed to load date models: {exc}'
            if DEBUG_OCR:
                print(f"[ocr] date recognizer disabled: {self._load_error}")
            return False

        self._tf = tf
        self._load_model = load_model
        self._ready = True
        return True

    def _normalize_label(self, label):
        return re.sub(r'[^a-z0-9]+', '', str(label or '').lower())

    def _crop_for_region(self, image, region):
        if image is None or region is None:
            return image, (0, 0)
        try:
            x1 = int(region.get('x1'))
            y1 = int(region.get('y1'))
            x2 = int(region.get('x2'))
            y2 = int(region.get('y2'))
        except Exception:
            return image, (0, 0)

        h, w = image.shape[:2]
        pad = max(4, int(min(w, h) * 0.03))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return image, (0, 0)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image, (0, 0)
        return crop, (x1, y1)

    def _detect_date_parts(self, image):
        if image is None or self.detector is None:
            return None, []

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None, []

        inference_img = cv2.resize(image, (512, 512))
        results = self.detector.predict(
            source=inference_img,
            conf=self.conf_threshold,
            imgsz=512,
            verbose=False
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None, []

        names = getattr(results[0], 'names', None) or getattr(self.detector, 'names', None) or {}
        scale_x = w / 512.0
        scale_y = h / 512.0

        boxes = []
        for box in results[0].boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)

            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
            cls = float(box.cls[0]) if hasattr(box, 'cls') else 0.0
            label = ''
            if isinstance(names, dict):
                label = names.get(int(cls), '')
            elif isinstance(names, (list, tuple)):
                idx = int(cls)
                if 0 <= idx < len(names):
                    label = names[idx]

            boxes.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'conf': conf,
                'cls': cls,
                'label': label,
                'cx': (x1 + x2) / 2.0
            })

        indicator = None
        parts = []
        for box in boxes:
            if self._normalize_label(box.get('label')) == 'indicator':
                if indicator is None or float(box.get('conf', 0.0)) > float(indicator.get('conf', 0.0)):
                    indicator = box
                continue
            parts.append(box)

        parts_sorted = sorted(parts, key=lambda item: item.get('cx', 0.0))
        return indicator, parts_sorted

    def _predict_day_month(self, image, box):
        if box is None or image is None:
            return '', 0.0
        x1, y1, x2, y2 = [int(box[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        if x2 <= x1 or y2 <= y1:
            return '', 0.0
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return '', 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(thresh, (64, 64))
        rgb_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        img_array = self._tf.keras.preprocessing.image.img_to_array(rgb_input)
        normalized = img_array / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)

        pred = self.day_month_model.predict(input_tensor, verbose=0)
        idx = int(np.argmax(pred))
        conf = float(np.max(pred))
        if idx < 0 or idx >= len(self.day_month_labels):
            return '', 0.0
        return self.day_month_labels[idx], conf

    def _predict_year(self, image, box):
        if box is None or image is None:
            return '', 0.0
        x1, y1, x2, y2 = [int(box[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        if x2 <= x1 or y2 <= y1:
            return '', 0.0
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return '', 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        resized = cv2.resize(thresh, (128, 32))
        normalized = resized.astype("float32") / 255.0
        input_tensor = np.expand_dims(normalized, axis=-1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        preds = self.year_model.predict(input_tensor, verbose=0)

        digits = []
        confs = []
        for pred in preds:
            if pred.ndim == 2:
                idx = int(np.argmax(pred, axis=1)[0])
                confs.append(float(np.max(pred[0])))
            else:
                idx = int(np.argmax(pred))
                confs.append(float(np.max(pred)))
            digits.append(idx)

        burmese_year = ''.join(DEFAULT_BURMESE_DIGITS[d] for d in digits if 0 <= d < len(DEFAULT_BURMESE_DIGITS))
        year_conf = float(np.mean(confs)) if confs else 0.0
        return burmese_year, year_conf

    def _box_to_region(self, box, label):
        if box is None:
            return None
        return {
            'x1': int(box.get('x1', 0)),
            'y1': int(box.get('y1', 0)),
            'x2': int(box.get('x2', 0)),
            'y2': int(box.get('y2', 0)),
            'conf': float(box.get('conf', 0.0)),
            'cls': float(box.get('cls', 0.0)),
            'label': label
        }

    def _synthesize_year_box(self, image, parts):
        if image is None or not parts:
            return None
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None

        sorted_parts = sorted(parts, key=lambda item: item.get('cx', 0.0))
        rightmost = sorted_parts[-1]
        y1 = min(int(p.get('y1', 0)) for p in sorted_parts)
        y2 = max(int(p.get('y2', 0)) for p in sorted_parts)
        x1 = int(rightmost.get('x2', 0))
        x2 = int(w)

        min_width = max(12, int(w * 0.18))
        if x2 - x1 < min_width:
            x1 = max(0, w - min_width)

        if x2 <= x1 or y2 <= y1:
            return None

        return {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'conf': 0.0,
            'cls': -1.0,
            'label': 'synthetic_year',
            'cx': (x1 + x2) / 2.0
        }

    def predict(self, image, region=None, label_prefix='date_of_birth'):
        if image is None:
            return {
                'birthDate': '',
                'birthDateLatin': '',
                'birthDateConfidence': 0.0,
                'regionBoxes': []
            }
        if not self._ensure_loaded():
            return {
                'birthDate': '',
                'birthDateLatin': '',
                'birthDateConfidence': 0.0,
                'regionBoxes': []
            }

        crop, offset = self._crop_for_region(image, region)
        indicator, parts = self._detect_date_parts(crop)
        if not parts and offset != (0, 0):
            crop = image
            offset = (0, 0)
            indicator, parts = self._detect_date_parts(crop)
        if not parts:
            return {
                'birthDate': '',
                'birthDateLatin': '',
                'birthDateConfidence': 0.0,
                'regionBoxes': []
            }

        day_box = parts[0] if len(parts) > 0 else None
        month_box = parts[1] if len(parts) > 1 else None
        year_box = parts[2] if len(parts) > 2 else None
        if year_box is None and len(parts) >= 2:
            year_box = self._synthesize_year_box(crop, parts)

        day_value, day_conf = self._predict_day_month(crop, day_box)
        month_value, month_conf = self._predict_day_month(crop, month_box)
        year_value, year_conf = self._predict_year(crop, year_box)

        birth_date = ''
        birth_date_latin = ''
        birth_conf = 0.0
        if day_value and month_value and year_value:
            birth_date = f"{day_value}.{month_value}.{year_value}"
            birth_date_latin = birth_date.translate(BURMESE_TO_LATIN)
            birth_conf = float(np.mean([day_conf, month_conf, year_conf]))

        region_boxes = []
        for label, box in (
            (f'{label_prefix}_day', day_box),
            (f'{label_prefix}_month', month_box),
            (f'{label_prefix}_year', year_box),
            (f'{label_prefix}_indicator', indicator)
        ):
            if box is None:
                continue
            region = self._box_to_region(box, label)
            if region:
                region['x1'] += offset[0]
                region['x2'] += offset[0]
                region['y1'] += offset[1]
                region['y2'] += offset[1]
                region_boxes.append(region)

        if day_box and month_box and year_box:
            xs = [day_box['x1'], month_box['x1'], year_box['x1']]
            ys = [day_box['y1'], month_box['y1'], year_box['y1']]
            xe = [day_box['x2'], month_box['x2'], year_box['x2']]
            ye = [day_box['y2'], month_box['y2'], year_box['y2']]
            union = {
                'x1': int(min(xs)) + offset[0],
                'y1': int(min(ys)) + offset[1],
                'x2': int(max(xe)) + offset[0],
                'y2': int(max(ye)) + offset[1],
                'conf': float(np.mean([day_box['conf'], month_box['conf'], year_box['conf']])),
                'cls': -1.0,
                'label': label_prefix
            }
            region_boxes.append(union)

        return {
            'birthDate': birth_date,
            'birthDateLatin': birth_date_latin,
            'birthDateConfidence': float(birth_conf),
            'regionBoxes': region_boxes
        }
