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
    DEFAULT_BURMESE_DIGITS,
    DEFAULT_CRNN_CHARSET,
    BURMESE_DIGITS,
    BURMESE_TO_LATIN,
    CRNN_NAING_PLACEHOLDER,
    CRNN_NAING_TOKEN
)
from core.preprocess import Preprocessor
from core.crnn import CRNN
from trocr_blood import read_blood_type

_recognizer = None
_lock = threading.Lock()

# python/core/recognizer.py
import os
from pathlib import Path
import json
DEBUG_OCR = os.getenv("OCR_DEBUG", "0") == "1"

AREA_MAP_PATH = Path(__file__).resolve().parents[2] / 'area_map.json'
BURMESE_DIGIT_SET = set(DEFAULT_BURMESE_DIGITS)
AREA_LINE_RE = re.compile(r'([၀-၉]{1,2})/([\u1000-\u109F]+)\(နိုင်\)')


def _load_area_map(path: Path):
    mapping = {}
    if not path.exists():
        return mapping
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict):
            for key, values in payload.items():
                if not isinstance(key, str):
                    continue
                if not isinstance(values, (list, tuple, set)):
                    continue
                cleaned = [v for v in values if isinstance(v, str) and v]
                if cleaned:
                    mapping[key] = set(cleaned)
    except Exception:
        return {}
    return mapping


_BURMESE_LETTER_RE = re.compile(r'^[\u1000-\u109F]+$')


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
        self.detector_names = getattr(self.detector, 'names', None) or getattr(self.detector.model, 'names', None) or {}
        self.area_detector = None
        self.area_names = {}
        if self.area_yolo_path and self.area_yolo_path.exists():
            self.area_detector = YOLO(str(self.area_yolo_path))
            self.area_names = getattr(self.area_detector, 'names', {}) or {}
        elif self.area_yolo_path:
            print(f'[ocr] area model not found: {self.area_yolo_path}')

        state = torch.load(str(self.crnn_path), map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}

        checkpoint_num_classes = None
        if isinstance(state, dict):
            fc_weight = state.get('fc.weight')
            if hasattr(fc_weight, 'shape'):
                checkpoint_num_classes = int(fc_weight.shape[0]) - 1

        charset = BURMESE_DIGITS
        if checkpoint_num_classes is not None and checkpoint_num_classes != len(BURMESE_DIGITS):
            if checkpoint_num_classes == len(DEFAULT_BURMESE_DIGITS):
                charset = DEFAULT_BURMESE_DIGITS
            elif checkpoint_num_classes == len(DEFAULT_CRNN_CHARSET):
                charset = DEFAULT_CRNN_CHARSET
            else:
                raise RuntimeError(
                    f'CRNN checkpoint expects {checkpoint_num_classes} classes, '
                    f'but OCR_CRNN_CLASSES_PATH provides {len(BURMESE_DIGITS)}. '
                    'Set OCR_CRNN_CLASSES_PATH / OCR_CRNN_WEIGHTS to a matching pair.'
                )
            print(
                f'[ocr] CRNN classes mismatch: checkpoint={checkpoint_num_classes}, '
                f'config={len(BURMESE_DIGITS)}. Using {len(charset)}-class charset.'
            )

        self.charset = charset
        self.crnn = CRNN(img_height=IMG_HEIGHT, num_classes=len(self.charset), hidden_size=HIDDEN_SIZE)
        self.crnn.load_state_dict(state, strict=True)
        self.crnn.to(self.device)
        self.crnn.eval()

        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        self.area_map = _load_area_map(AREA_MAP_PATH)
        # python/core/recognizer.py inside NRCRecognizer.__init__
        if DEBUG_OCR:
            print(f"[ocr] charset len={len(self.charset)} first10={self.charset[:10]}")


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
        if CRNN_NAING_PLACEHOLDER in raw_digits:
            raw_digits = raw_digits.replace(CRNN_NAING_PLACEHOLDER, CRNN_NAING_TOKEN)
        nrc_burmese = raw_digits
        nrc_latin = raw_digits.translate(BURMESE_TO_LATIN) if raw_digits else ''

        conf = float(np.mean([b[4] for b in boxes])) if len(boxes) else 0.0
        elapsed = (time.time() - start) * 1000

        # python/core/recognizer.py inside NRCRecognizer.recognize

        if DEBUG_OCR:
            print(f"[ocr] image shape={image.shape}")

        region_boxes = self._detect_regions(image, AREA_CONF_THRESHOLD)
        if DEBUG_OCR:
            print(f"[ocr] region_boxes={len(region_boxes)}")

        preprocessed = self.preprocessor.preprocess_full_image(image)
        boxes = self._detect_boxes(preprocessed, conf_threshold)

        if DEBUG_OCR:
            print(f"[ocr] yolo boxes={len(boxes)}")
            if len(boxes):
                print(f"[ocr] box confs={[round(b[4], 3) for b in boxes]}")
                print(f"[ocr] box xyxy={[list(map(int, b[:4])) for b in boxes]}")

        digit_crops = self._crop_digits(image, boxes)
        if DEBUG_OCR:
            print(f"[ocr] digit_crops={len(digit_crops)} sizes={[c.shape[:2] for c in digit_crops]}")

        corrected_burmese = self._correct_nrc_from_boxes(boxes, region_boxes)
        if corrected_burmese:
            nrc_burmese = corrected_burmese
            nrc_latin = corrected_burmese.translate(BURMESE_TO_LATIN)

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

        def _infer_from_zones(zones, debug_prefix):
            conf_sums = defaultdict(float)
            counts = defaultdict(int)
            variant_conf_sums = {0: defaultdict(float), 1: defaultdict(float)}
            variant_counts = {0: defaultdict(int), 1: defaultdict(int)}
            ab_hits = 0
            ab_conf_sum = 0.0

            for zone_idx, zone in enumerate(zones):
                zone = self._upscale_for_trocr(zone)
                for variant_idx, variant in enumerate(self._build_blood_variants(zone)):
                    pil_image = Image.fromarray(variant)
                    try:
                        text, confidence = read_blood_type(pil_image)
                        print(
                            f"[DEBUG] Blood OCR {debug_prefix}{zone_idx} v{variant_idx} "
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
                    conf_sums[cls] += c
                    counts[cls] += 1
                    if variant_idx in (0, 1):
                        variant_conf_sums[variant_idx][cls] += c
                        variant_counts[variant_idx][cls] += 1

            if ab_hits >= 2:
                ab_avg_conf = ab_conf_sum / ab_hits
                if ab_avg_conf >= 0.72:
                    return 'အေဘီ', float(ab_avg_conf)

            # A confirmation for fallback zones: if RGB repeatedly predicts strong A, prefer A.
            a_v0_count = variant_counts[0].get('အေ', 0)
            if a_v0_count >= 1:
                a_v0_avg = variant_conf_sums[0].get('အေ', 0.0) / a_v0_count
                if a_v0_avg >= 0.78:
                    return 'အေ', float(a_v0_avg)

            # Stricter O safeguard: require strong RGB O plus supporting threshold O evidence.
            o_v0_count = variant_counts[0].get('အို', 0)
            o_v1_count = variant_counts[1].get('အို', 0)
            b_v0_count = variant_counts[0].get('ဘီ', 0)
            if o_v0_count > 0 and o_v1_count > 0:
                o_v0_avg = variant_conf_sums[0].get('အို', 0.0) / o_v0_count
                o_v1_avg = variant_conf_sums[1].get('အို', 0.0) / o_v1_count
                if (
                    o_v0_count >= 2
                    and o_v0_avg >= 0.97
                    and o_v1_avg >= 0.88
                    and o_v0_count > b_v0_count
                ):
                    return 'အို', float((o_v0_avg + o_v1_avg) / 2.0)

            count_margin = 2
            conf_margin = 0.08

            def _avg(conf_map, count_map, key):
                c = count_map.get(key, 0)
                return (conf_map.get(key, 0.0) / c) if c > 0 else 0.0

            def _decide_ab(count_map, conf_map):
                a_count = count_map.get('အေ', 0)
                b_count = count_map.get('ဘီ', 0)
                if a_count == 0 and b_count == 0:
                    return None
                a_avg = _avg(conf_map, count_map, 'အေ')
                b_avg = _avg(conf_map, count_map, 'ဘီ')

                if a_count > 0 and b_count == 0:
                    return 'အေ', a_avg
                if b_count > 0 and a_count == 0:
                    return 'ဘီ', b_avg

                count_gap = a_count - b_count
                if count_gap >= count_margin:
                    return 'အေ', a_avg
                if count_gap <= -count_margin:
                    return 'ဘီ', b_avg

                conf_gap = a_avg - b_avg
                if conf_gap >= conf_margin:
                    return 'အေ', a_avg
                if conf_gap <= -conf_margin:
                    return 'ဘီ', b_avg
                return None

            decision = _decide_ab(variant_counts[0], variant_conf_sums[0])
            if decision is None:
                decision = _decide_ab(variant_counts[1], variant_conf_sums[1])
            if decision is not None:
                return decision[0], float(decision[1])

            has_a_or_b = (counts.get('အေ', 0) > 0) or (counts.get('ဘီ', 0) > 0)
            if has_a_or_b:
                return '', 0.0

            o_count = counts.get('အို', 0)
            if o_count > 0:
                o_avg = conf_sums.get('အို', 0.0) / o_count
                return 'အို', float(o_avg)
            return '', 0.0

        # Primary: one single right-side crop after label.
        # Option 1: if v0/v1 disagree, pick the higher-confidence one.
        primary_zone = self._extract_primary_blood_value_zone(crop)
        primary_zone = self._upscale_for_trocr(primary_zone)
        primary_reads = []
        for variant_idx, variant in enumerate(self._build_blood_variants(primary_zone)):
            pil_image = Image.fromarray(variant)
            try:
                text, confidence = read_blood_type(pil_image)
                print(
                    f"[DEBUG] Blood OCR p0 v{variant_idx} "
                    f"({primary_zone.shape[1]}x{primary_zone.shape[0]}): '{text}' (confidence: {confidence:.2f})"
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

            primary_reads.append((cls, float(confidence), variant_idx))

        if primary_reads:
            primary_by_variant = {}
            for cls, conf, v_idx in primary_reads:
                primary_by_variant[v_idx] = (cls, conf)

            # A priority from RGB in primary crop: trust v0 A unless there is
            # very strong contradictory evidence.
            p_v0 = primary_by_variant.get(0)
            p_v1 = primary_by_variant.get(1)
            if p_v0:
                p0_cls, p0_conf = p_v0
                if p0_cls == 'အေ' and p0_conf >= 0.74:
                    if not p_v1:
                        return 'အေ', float(p0_conf), target
                    p1_cls, p1_conf = p_v1
                    strong_contradiction = (
                        (p1_cls == 'အေဘီ' and p1_conf >= 0.90) or
                        (p1_cls == 'အို' and p1_conf >= 0.96)
                    )
                    if not strong_contradiction:
                        return 'အေ', float(p0_conf), target

            # AB is often under-confident in this model; prioritize it when present.
            primary_ab_confs = [conf for cls, conf, _ in primary_reads if cls == 'အေဘီ']
            PRIMARY_AB_MIN_CONF = 0.80
            primary_ab_hits = sum(1 for cls, _, _ in primary_reads if cls == 'အေဘီ')
            if primary_ab_confs and (primary_ab_hits >= 2 or max(primary_ab_confs) >= 0.90):
                best_ab_conf = max(primary_ab_confs)
                if best_ab_conf >= PRIMARY_AB_MIN_CONF:
                    return 'အေဘီ', float(best_ab_conf), target

            # A confirmation for primary crop:
            # prefer A when RGB sees confident A, unless threshold strongly contradicts it.
            if p_v0 and p_v1:
                p0_cls, p0_conf = p_v0
                p1_cls, p1_conf = p_v1
                if p0_cls == 'အေ' and p0_conf >= 0.82 and (p1_conf - p0_conf) < 0.12:
                    return 'အေ', float(p0_conf), target

            # If same class appears in both variants, merge confidence.
            if p_v0 and p_v1 and p_v0[0] == p_v1[0]:
                merged_conf = (p_v0[1] + p_v1[1]) / 2.0
                return p_v0[0], float(merged_conf), target

            # If variants disagree:
            # - choose higher-confidence only when margin is clear
            # - otherwise fall back to zone strategy
            if p_v0 and p_v1:
                sorted_reads = sorted((p_v0, p_v1), key=lambda it: it[1], reverse=True)
                best_cls, best_conf = sorted_reads[0]
                second_cls, second_conf = sorted_reads[1]
                DISAGREE_CONF_MARGIN = 0.08
                if (best_conf - second_conf) >= DISAGREE_CONF_MARGIN:
                    return best_cls, float(best_conf), target
            else:
                # Only one valid primary read, use it directly.
                only_cls, only_conf, _only_v = primary_reads[0]
                return only_cls, float(only_conf), target

        # Fallback: old multi-zone strategy.
        blood_type, blood_conf = _infer_from_zones(self._extract_blood_value_zones(crop), 'z')
        if blood_type:
            return blood_type, blood_conf, target
        return '', 0.0, target

    def _extract_primary_blood_value_zone(self, crop):
        h, w = crop.shape[:2]
        x0 = int(w * 0.55)
        y0 = int(h * 0.10)
        y1 = int(h * 0.90)
        x0 = max(0, min(x0, w - 1))
        if y1 <= y0:
            y0, y1 = 0, h
        primary = crop[y0:y1, x0:]
        if primary.size == 0:
            primary = crop[:, x0:]
        if primary.size == 0:
            primary = crop
        return primary

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

    def _normalize_region_label(self, value):
        return re.sub(r'[^a-z0-9]+', '', str(value or '').lower())

    def _pick_nrc_region(self, regions):
        if not regions:
            return None
        candidates = []
        for region in regions:
            label = self._normalize_region_label(region.get('label'))
            if label in {'nrcnumber', 'nrcnum', 'nrcno'} or label.startswith('nrcnumber'):
                candidates.append(region)
        if not candidates:
            return None
        return max(candidates, key=lambda r: float(r.get('conf', 0.0)))

    def _cls_to_label(self, cls_value):
        if cls_value is None:
            return ''
        try:
            idx = int(round(float(cls_value)))
        except Exception:
            return ''
        label = None
        if isinstance(self.detector_names, dict):
            label = self.detector_names.get(idx)
        elif isinstance(self.detector_names, (list, tuple)):
            if 0 <= idx < len(self.detector_names):
                label = self.detector_names[idx]
        if label is None:
            return ''

        label = str(label)
        if label.startswith('digit_'):
            suffix = label.split('_', 1)[-1]
            if suffix.isdigit():
                digit_idx = int(suffix)
                if 0 <= digit_idx < len(DEFAULT_BURMESE_DIGITS):
                    return DEFAULT_BURMESE_DIGITS[digit_idx]

        if label in DEFAULT_BURMESE_DIGITS:
            return label
        if 'နိုင်' in label or label == CRNN_NAING_TOKEN:
            return CRNN_NAING_TOKEN
        if _BURMESE_LETTER_RE.match(label):
            return label
        return ''

    def _ordered_labels_from_boxes(self, boxes, region_boxes):
        if boxes is None or len(boxes) == 0:
            return []
        region = self._pick_nrc_region(region_boxes)
        items = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if region:
                if not (region['x1'] <= cx <= region['x2'] and region['y1'] <= cy <= region['y2']):
                    continue
            label = self._cls_to_label(cls)
            if not label:
                continue
            items.append({
                'label': label,
                'conf': float(conf),
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'cx': float(cx)
            })
        items.sort(key=lambda item: (item['x1'], item['y1']))
        return items

    def _extract_prefix(self, labels):
        digits = []
        last_digit_index = -1
        for idx, item in enumerate(labels):
            label = item['label']
            if label in BURMESE_DIGIT_SET:
                if len(digits) < 2:
                    digits.append(label)
                    last_digit_index = idx
                else:
                    break
                continue
            if digits:
                break
        prefix = ''.join(digits)
        return prefix, last_digit_index

    def _score_code(self, code, labels, start_index):
        if not code:
            return None
        score = 0.0
        matched = 0
        for item in labels[start_index + 1:]:
            if matched >= len(code):
                break
            if item['label'] == code[matched]:
                score += item['conf']
                matched += 1
        if matched != len(code):
            return None
        return score

    def _extract_area_letters(self, labels, start_index, region_boxes):
        letters = []
        region = self._pick_nrc_region(region_boxes)
        region_min = float(region['x1']) if region else None
        region_max = float(region['x2']) if region else None

        letter_items = []
        for item in labels[start_index + 1:]:
            ch = item.get('label', '')
            if ch == CRNN_NAING_TOKEN:
                continue
            if ch in BURMESE_DIGIT_SET:
                continue
            if not _BURMESE_LETTER_RE.match(ch):
                continue
            letter_items.append(item)

        left_boundary = None
        right_boundary = None
        if 0 <= start_index < len(labels):
            left_boundary = labels[start_index].get('x2') or labels[start_index].get('cx')

        naing_item = None
        for item in labels[start_index + 1:]:
            if item.get('label') == CRNN_NAING_TOKEN:
                naing_item = item
                break
        if naing_item is not None:
            right_boundary = naing_item.get('x1') or naing_item.get('cx')

        if left_boundary is None and region_min is not None:
            left_boundary = region_min
        if right_boundary is None and region_max is not None:
            right_boundary = region_max

        min_cx = max_cx = None
        if left_boundary is not None and right_boundary is not None and right_boundary > left_boundary:
            min_cx = left_boundary
            max_cx = right_boundary
        elif len(letter_items) >= 2:
            min_cx = min(item['cx'] for item in letter_items if item.get('cx') is not None)
            max_cx = max(item['cx'] for item in letter_items if item.get('cx') is not None)
        elif len(letter_items) == 1 and region_min is not None and region_max is not None:
            min_cx = region_min
            max_cx = region_max

        def _position_from_cx(cx):
            if min_cx is None or max_cx is None or max_cx <= min_cx:
                return None
            span = max_cx - min_cx
            if cx <= min_cx + span / 3:
                return 0
            if cx <= min_cx + 2 * span / 3:
                return 1
            return 2

        for item in letter_items:
            pos = _position_from_cx(item.get('cx'))
            letters.append((item['label'], float(item['conf']), pos, item.get('cx')))
            if len(letters) >= 3:
                break

        if len(letters) >= 2:
            pos_hints = [pos if pos in (0, 1, 2) else None for _ch, _conf, pos, _cx in letters]
            has_duplicates = len(set([p for p in pos_hints if p is not None])) != len([p for p in pos_hints if p is not None])
            if has_duplicates or any(p is None for p in pos_hints):
                letters_sorted = sorted(letters, key=lambda it: (it[3] is None, it[3] if it[3] is not None else 0))
                reassigned = []
                for idx, (ch, conf, _pos, cx) in enumerate(letters_sorted[:3]):
                    reassigned.append((ch, conf, idx, cx))
                letters = reassigned
        return letters

    def _extract_suffix_digits(self, labels, start_index):
        if not labels:
            return ''
        naing_index = None
        for idx in range(start_index + 1, len(labels)):
            if labels[idx].get('label') == CRNN_NAING_TOKEN:
                naing_index = idx
                break
        if naing_index is None:
            return ''
        digits = []
        for item in labels[naing_index + 1:]:
            ch = item.get('label')
            if ch in BURMESE_DIGIT_SET:
                digits.append(ch)
        return ''.join(digits)

    def _map_letters_to_positions(self, observed, candidates):
        mapped = []
        remaining = set(candidates)
        for ch, conf, pos_hint, _cx in observed:
            matched = False
            if pos_hint not in (0, 1, 2):
                print(f"[ocr] nrc eliminate skip letter={ch} (no position)")
                continue
            positions = (pos_hint,)
            for pos in positions:
                next_remaining = {c for c in remaining if len(c) > pos and c[pos] == ch}
                print(f"[ocr] nrc eliminate try letter={ch} pos={pos+1} remaining={len(next_remaining)}")
                if next_remaining:
                    remaining = next_remaining
                    mapped.append((pos, ch, conf))
                    print(f"[ocr] nrc eliminate use letter={ch} pos={pos+1} remaining={len(remaining)}")
                    matched = True
                    break
            if not matched:
                print(f"[ocr] nrc eliminate skip letter={ch} (no matches)")

        # If only one letter is detected and it landed on the rightmost position,
        # exclude candidates that have that same rightmost letter (user rule).
        if len(observed) == 1 and mapped and mapped[0][0] == 2:
            ch = mapped[0][1]
            next_remaining = {c for c in remaining if len(c) <= 2 or c[2] != ch}
            print(f"[ocr] nrc eliminate exclude rightmost={ch} remaining={len(next_remaining)}")
            if next_remaining:
                remaining = next_remaining
        return mapped, remaining

    def _correct_nrc_from_boxes(self, boxes, region_boxes):
        if not self.area_map:
            return ''
        labels = self._ordered_labels_from_boxes(boxes, region_boxes)
        if not labels:
            print("[ocr] nrc labels=EMPTY")
            return ''
        prefix, last_digit_index = self._extract_prefix(labels)
        if not prefix:
            ordered_labels = ''.join([item['label'] for item in labels])
            print(f"[ocr] nrc labels={ordered_labels}")
            print("[ocr] nrc prefix=EMPTY")
            return ''
        candidates = None
        if len(prefix) >= 2 and prefix[:2] in self.area_map:
            candidates = self.area_map.get(prefix[:2])
            prefix = prefix[:2]
        elif prefix[:1] in self.area_map:
            candidates = self.area_map.get(prefix[:1])
            prefix = prefix[:1]
        if not candidates:
            ordered_labels = ''.join([item['label'] for item in labels])
            print(f"[ocr] nrc labels={ordered_labels}")
            print(f"[ocr] nrc prefix={prefix} candidates=EMPTY")
            return ''

        # Eliminate candidates based on readable area letters (left->right).
        observed_letters = self._extract_area_letters(labels, last_digit_index, region_boxes)

        print(f"[ocr] nrc eliminate start count={len(candidates)} candidates={sorted(candidates)}")
        observed_dump = ','.join([f"{ch}:{pos+1 if pos is not None else '?'}:{conf:.2f}" for ch, conf, pos, _cx in observed_letters]) or 'EMPTY'
        print(f"[ocr] nrc observed letters={observed_dump}")
        if observed_letters:
            direct_code = ''.join([ch for ch, _conf, _pos, _cx in observed_letters])[:3]
            if len(direct_code) == 3 and direct_code in candidates:
                print(f"[ocr] nrc direct match={direct_code}")
                candidates = {direct_code}

        mapped_letters, filtered = self._map_letters_to_positions(observed_letters, candidates)
        candidates = filtered

        best_code = ''
        best_score = 0.0
        best_mismatches = None
        for code in candidates:
            if not code:
                continue
            mismatches = 0
            score = 0.0
            for pos, ch, conf in mapped_letters:
                if pos >= len(code) or code[pos] != ch:
                    mismatches += 1
                else:
                    score += conf
            if best_mismatches is None or mismatches < best_mismatches:
                best_mismatches = mismatches
                best_score = score
                best_code = code
            elif mismatches == best_mismatches and score > best_score:
                best_score = score
                best_code = code

        ordered_labels = ''.join([item['label'] for item in labels])
        print(f"[ocr] nrc labels={ordered_labels}")
        print(f"[ocr] nrc prefix={prefix} candidates={sorted(candidates)}")
        if not best_code:
            print("[ocr] nrc best=EMPTY")
            return ''
        print(f"[ocr] nrc best={best_code} score={round(best_score, 3)} mismatches={best_mismatches}")

        suffix_digits = self._extract_suffix_digits(labels, last_digit_index)
        return f"{prefix}/{best_code}(နိုင်){suffix_digits}"

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
