import threading
import time
import re
from collections import defaultdict
import numpy as np
import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from core.config import (
    YOLO_WEIGHTS,
    AREA_YOLO_WEIGHTS,
    CRNN_WEIGHTS,
    IMG_HEIGHT,
    IMG_WIDTH,
    HIDDEN_SIZE,
    CONF_THRESHOLD,
    AREA_CONF_THRESHOLD,
    BURMESE_DIGITS,
    BURMESE_TO_LATIN
)
from core.preprocess import Preprocessor
from core.crnn import CRNN
from trocr_blood import read_blood_type

_recognizer = None
_lock = threading.Lock()


class NRCRecognizer:
    """End-to-end NRC recognition pipeline"""

    def __init__(self, yolo_weights, crnn_weights, area_weights, device=None):
        if not yolo_weights.exists():
            raise FileNotFoundError(f'YOLO weights not found: {yolo_weights}')
        if not crnn_weights.exists():
            raise FileNotFoundError(f'CRNN weights not found: {crnn_weights}')

        self.yolo_path = yolo_weights
        self.area_yolo_path = area_weights
        self.crnn_path = crnn_weights
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocessor = Preprocessor(target_height=IMG_HEIGHT, target_width=IMG_WIDTH)
        self.detector = YOLO(str(self.yolo_path))
        self.area_detector = None
        self.area_names = {}
        if self.area_yolo_path and self.area_yolo_path.exists():
            self.area_detector = YOLO(str(self.area_yolo_path))
            self.area_names = getattr(self.area_detector, 'names', {}) or {}
        elif self.area_yolo_path:
            print(f'[ocr] area model not found: {self.area_yolo_path}')

        self.crnn = CRNN(img_height=IMG_HEIGHT, num_classes=len(BURMESE_DIGITS), hidden_size=HIDDEN_SIZE)
        state = torch.load(str(self.crnn_path), map_location=self.device)
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        self.crnn.load_state_dict(state, strict=True)
        self.crnn.to(self.device)
        self.crnn.eval()

        self.idx_to_char = {idx: char for idx, char in enumerate(BURMESE_DIGITS)}

    def recognize(self, image, conf_threshold=CONF_THRESHOLD):
        start = time.time()

        region_boxes = self._detect_regions(image, AREA_CONF_THRESHOLD)
        blood_type, blood_type_conf, blood_type_box = self._recognize_blood_type(image, region_boxes)
        preprocessed = self.preprocessor.preprocess_full_image(image)
        boxes = self._detect_boxes(preprocessed, conf_threshold)

        if len(boxes) == 0:
            return self._empty_result(start, [], region_boxes, blood_type, blood_type_conf, blood_type_box)

        digit_crops = self._crop_digits(image, boxes)
        if not digit_crops:
            return self._empty_result(start, boxes, region_boxes, blood_type, blood_type_conf, blood_type_box)

        concat_image = self._concat_crops(digit_crops)
        processed = self.preprocessor.preprocess_for_crnn(concat_image)

        image_tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.crnn(image_tensor)

        raw_digits = self._ctc_decode(output)
        nrc_burmese = raw_digits
        nrc_latin = raw_digits.translate(BURMESE_TO_LATIN) if raw_digits else ''

        conf = float(np.mean([b[4] for b in boxes])) if len(boxes) else 0.0
        elapsed = (time.time() - start) * 1000

        return {
            'nrcNumber': nrc_latin,
            'nrcNumberBurmese': nrc_burmese,
            'rawDigits': raw_digits,
            'confidence': conf,
            'bloodType': blood_type,
            'bloodTypeConfidence': blood_type_conf,
            'bloodTypeBox': blood_type_box,
            'boxes': [self._box_to_dict(b) for b in boxes],
            'regionBoxes': region_boxes,
            'inferenceMs': round(elapsed, 2),
            'model': {
                'detector': self.yolo_path.name,
                'recognizer': self.crnn_path.name,
                'areaDetector': self.area_yolo_path.name if self.area_detector else None
            }
        }

    def _detect_boxes(self, image, conf_threshold):
        results = self.detector(image, conf=conf_threshold, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        boxes = np.column_stack((boxes_xyxy, confs, classes))
        boxes = boxes[boxes[:, 0].argsort()]
        return boxes

    def _crop_digits(self, image, boxes):
        digit_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                digit_crops.append(crop)
        return digit_crops

    def _concat_crops(self, crops):
        target_h = IMG_HEIGHT
        resized_crops = []
        for crop in crops:
            h, w = crop.shape[:2]
            if h == 0 or w == 0:
                continue
            new_w = max(1, int(w * target_h / h))
            resized = cv2.resize(crop, (new_w, target_h))
            resized_crops.append(resized)

        if not resized_crops:
            return crops[0]

        return np.hstack(resized_crops)

    def _detect_regions(self, image, conf_threshold):
        if self.area_detector is None:
            return []

        results = self.area_detector(image, conf=conf_threshold, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        names = getattr(results[0], 'names', None) or self.area_names or {}

        regions = []
        for box_xyxy, conf, cls in zip(boxes_xyxy, confs, classes):
            label = None
            if isinstance(names, dict):
                label = names.get(int(cls))
            elif isinstance(names, (list, tuple)):
                idx = int(cls)
                if 0 <= idx < len(names):
                    label = names[idx]
            if not label:
                label = f'region_{int(cls)}'
            regions.append(self._region_box_to_dict(box_xyxy, conf, cls, label))

        return regions

    def _recognize_blood_type(self, image, region_boxes):
        if image is None or not region_boxes:
            return '', 0.0, None

        candidates = []
        for box in region_boxes:
            label = (box.get('label') or '').strip()
            normalized = re.sub(r'[^a-z0-9]+', '', label.lower())
            if normalized in {'bloodtype', 'bloodgroup'} or 'blood' in normalized:
                candidates.append(box)

        if not candidates:
            return '', 0.0, None

        target = max(candidates, key=lambda b: float(b.get('conf', 0.0)))

        h, w = image.shape[:2]
        pad = 4
        x1 = max(0, int(target['x1']) - pad)
        y1 = max(0, int(target['y1']) - pad)
        # Do not expand too far right; detector box already covers blood_type field.
        x2 = min(w, int(target['x2']) + pad)
        y2 = min(h, int(target['y2']) + pad)

        if x2 <= x1 or y2 <= y1:
            return '', 0.0, target

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return '', 0.0, target

        votes = defaultdict(float)
        conf_sums = defaultdict(float)
        counts = defaultdict(int)
        ab_hits = 0
        ab_conf_sum = 0.0

        for zone_idx, zone in enumerate(self._extract_blood_value_zones(crop)):
            zone = self._upscale_for_trocr(zone)
            for variant_idx, variant in enumerate(self._build_blood_variants(zone)):
                pil_image = Image.fromarray(variant)
                try:
                    text, confidence = read_blood_type(pil_image)
                    print(
                        f"[DEBUG] Blood OCR z{zone_idx} v{variant_idx} "
                        f"({zone.shape[1]}x{zone.shape[0]}): '{text}' (confidence: {confidence:.2f})"
                    )
                except Exception as exc:
                    print(f'[ocr] blood type OCR failed: {exc}')
                    continue

                cleaned = self._normalize_blood_type(text)
                if not cleaned or self._looks_like_blood_label(cleaned):
                    continue

                cls = self._canonical_blood_class(cleaned)
                if not cls:
                    continue

                c = float(confidence)
                if cls == 'အေဘီ':
                    ab_hits += 1
                    ab_conf_sum += c
                # Give stronger weight to later (more-right) zones to avoid label text on left.
                right_bias = 1.0 + (0.25 * zone_idx)
                votes[cls] += right_bias * (1.0 + (0.10 * c))
                conf_sums[cls] += c
                counts[cls] += 1

        # 1) If AB appears with reasonable confidence, prefer it.
        if ab_hits >= 1:
            ab_avg_conf = (ab_conf_sum / ab_hits) if ab_hits else 0.0
            if ab_avg_conf >= 0.65:
                return 'အေဘီ', float(ab_avg_conf), target

        # O vs B rule:
        # - if O appears with stable ~0.97 confidence, prefer O
        # - otherwise choose lower average confidence between O and B
        o_count = counts.get('အို', 0)
        b_count = counts.get('ဘီ', 0)
        if o_count > 0 and b_count > 0:
            o_avg = conf_sums.get('အို', 0.0) / max(1, o_count)
            b_avg = conf_sums.get('ဘီ', 0.0) / max(1, b_count)
            o_is_stable_097 = abs(o_avg - 0.97) <= 0.01
            if o_is_stable_097:
                return 'အို', float(o_avg), target
            if o_avg <= b_avg:
                return 'အို', float(o_avg), target
            return 'ဘီ', float(b_avg), target

        # 2) Compare only A vs B.
        a_count = counts.get('အေ', 0)
        b_count = counts.get('ဘီ', 0)
        if a_count > 0 or b_count > 0:
            a_avg = conf_sums.get('အေ', 0.0) / max(1, a_count)
            b_avg = conf_sums.get('ဘီ', 0.0) / max(1, b_count)

            # Requested heuristic: for A vs B, choose the lower average confidence.
            if a_count > 0 and b_count > 0:
                if a_avg <= b_avg:
                    return 'အေ', float(a_avg), target
                return 'ဘီ', float(b_avg), target

            # If only one exists, return that one.
            if a_count > 0:
                return 'အေ', float(a_avg), target
            return 'ဘီ', float(b_avg), target

        # 3) Use O only as fallback when no A/B/AB evidence exists.
        if counts.get('အို', 0) > 0:
            o_avg = conf_sums.get('အို', 0.0) / max(1, counts.get('အို', 0))
            return 'အို', float(o_avg), target

        return '', 0.0, target

    def _extract_blood_value_zones(self, crop):
        h, w = crop.shape[:2]
        zones = []
        # Use only far-right zones; left part is usually the field label.
        # Include a bit more left so AB (အေဘီ) is not truncated to just ဘီ.
        for start_ratio in (0.55, 0.62, 0.70, 0.78):
            x0 = int(w * start_ratio)
            if w - x0 > 12:
                zones.append(crop[:, x0:])

        y0 = int(h * 0.1)
        y1 = int(h * 0.9)
        if y1 - y0 > 8:
            center = crop[y0:y1, :]
            for start_ratio in (0.55, 0.62, 0.70, 0.78):
                x0 = int(center.shape[1] * start_ratio)
                if center.shape[1] - x0 > 12:
                    zones.append(center[:, x0:])

        if not zones:
            zones.append(crop)

        return zones

    def _build_blood_variants(self, crop):
        # Match your standalone script first (RGB), then add threshold variant.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return (
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB),
        )

    def _upscale_for_trocr(self, crop):
        h, w = crop.shape[:2]
        max_side = max(h, w)
        if max_side < 160:
            scale = 3
        elif max_side < 240:
            scale = 2
        else:
            scale = 1
        if scale == 1:
            return crop
        return cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    def _normalize_blood_type(self, text):
        if not text:
            return ''
        return re.sub(r'\s+', '', str(text))

    def _canonical_blood_class(self, value):
        v = value.strip()
        if not v:
            return ''
        u = v.upper()
        if 'အေဘီ' in v or 'ေအဘီ' in v or 'AB' in u:
            return 'အေဘီ'
        if 'အေ' in v or 'ေအ' in v or (u == 'A'):
            return 'အေ'
        if 'ဘီ' in v or (u == 'B'):
            return 'ဘီ'
        # Keep O last to avoid dominating noisy reads.
        if 'အို' in v or 'အုိ' in v or u in {'O', '0'}:
            return 'အို'
        return ''

    def _looks_like_blood_label(self, value):
        return (
            'သွေး' in value or
            'အမျိုး' in value or
            'အစား' in value or
            'blood' in value.lower()
        )

    def _ctc_decode(self, output):
        output = output.squeeze(1)
        _, preds = output.max(dim=1)
        preds = preds.cpu().numpy()

        decoded = []
        prev = None
        blank_idx = len(self.idx_to_char)

        for p in preds:
            if p != blank_idx and p != prev:
                decoded.append(self.idx_to_char[p])
            prev = p

        return ''.join(decoded)

    def _box_to_dict(self, box):
        return {
            'x1': int(box[0]),
            'y1': int(box[1]),
            'x2': int(box[2]),
            'y2': int(box[3]),
            'conf': float(box[4]),
            'cls': float(box[5])
        }

    def _region_box_to_dict(self, box_xyxy, conf, cls, label):
        return {
            'x1': int(box_xyxy[0]),
            'y1': int(box_xyxy[1]),
            'x2': int(box_xyxy[2]),
            'y2': int(box_xyxy[3]),
            'conf': float(conf),
            'cls': float(cls),
            'label': label
        }

    def _empty_result(self, start_time, boxes, region_boxes, blood_type='', blood_type_conf=0.0, blood_type_box=None):
        elapsed = (time.time() - start_time) * 1000
        return {
            'nrcNumber': '',
            'nrcNumberBurmese': '',
            'rawDigits': '',
            'confidence': 0.0,
            'bloodType': blood_type,
            'bloodTypeConfidence': float(blood_type_conf),
            'bloodTypeBox': blood_type_box,
            'boxes': [self._box_to_dict(b) for b in boxes] if boxes else [],
            'regionBoxes': region_boxes if region_boxes else [],
            'inferenceMs': round(elapsed, 2),
            'model': {
                'detector': self.yolo_path.name,
                'recognizer': self.crnn_path.name,
                'areaDetector': self.area_yolo_path.name if self.area_detector else None
            }
        }


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        with _lock:
            if _recognizer is None:
                _recognizer = NRCRecognizer(YOLO_WEIGHTS, CRNN_WEIGHTS, AREA_YOLO_WEIGHTS)
    return _recognizer
