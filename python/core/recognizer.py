import threading
import time
import numpy as np
import torch
import cv2
from ultralytics import YOLO

from core.config import (
    YOLO_WEIGHTS,
    CRNN_WEIGHTS,
    IMG_HEIGHT,
    IMG_WIDTH,
    HIDDEN_SIZE,
    CONF_THRESHOLD,
    BURMESE_DIGITS,
    BURMESE_TO_LATIN
)
from core.preprocess import Preprocessor
from core.crnn import CRNN

_recognizer = None
_lock = threading.Lock()

class NRCRecognizer:
    """End-to-end NRC recognition pipeline"""

    def __init__(self, yolo_weights, crnn_weights, device=None):
        if not yolo_weights.exists():
            raise FileNotFoundError(f'YOLO weights not found: {yolo_weights}')
        if not crnn_weights.exists():
            raise FileNotFoundError(f'CRNN weights not found: {crnn_weights}')

        self.yolo_path = yolo_weights
        self.crnn_path = crnn_weights
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocessor = Preprocessor(target_height=IMG_HEIGHT, target_width=IMG_WIDTH)
        self.detector = YOLO(str(self.yolo_path))

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

        preprocessed = self.preprocessor.preprocess_full_image(image)
        boxes = self._detect_boxes(preprocessed, conf_threshold)

        if len(boxes) == 0:
            return self._empty_result(start, [])

        digit_crops = self._crop_digits(image, boxes)
        if not digit_crops:
            return self._empty_result(start, boxes)

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
            'boxes': [self._box_to_dict(b) for b in boxes],
            'inferenceMs': round(elapsed, 2),
            'model': {
                'detector': self.yolo_path.name,
                'recognizer': self.crnn_path.name
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

    def _empty_result(self, start_time, boxes):
        elapsed = (time.time() - start_time) * 1000
        return {
            'nrcNumber': '',
            'nrcNumberBurmese': '',
            'rawDigits': '',
            'confidence': 0.0,
            'boxes': [self._box_to_dict(b) for b in boxes] if boxes else [],
            'inferenceMs': round(elapsed, 2),
            'model': {
                'detector': self.yolo_path.name,
                'recognizer': self.crnn_path.name
            }
        }


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        with _lock:
            if _recognizer is None:
                _recognizer = NRCRecognizer(YOLO_WEIGHTS, CRNN_WEIGHTS)
    return _recognizer
