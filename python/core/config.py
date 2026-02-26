from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv('OCR_MODELS_DIR', ROOT_DIR / 'models')).resolve()

YOLO_WEIGHTS = Path(os.getenv('OCR_YOLO_WEIGHTS', MODELS_DIR / 'best.pt')).resolve()
AREA_YOLO_WEIGHTS = Path(os.getenv('OCR_AREA_YOLO_WEIGHTS', MODELS_DIR / 'best_area.pt')).resolve()
CRNN_WEIGHTS = Path(os.getenv('OCR_CRNN_WEIGHTS', MODELS_DIR / 'crnn.pth')).resolve()

IMG_HEIGHT = int(os.getenv('OCR_IMG_HEIGHT', '32'))
IMG_WIDTH = int(os.getenv('OCR_IMG_WIDTH', '128'))
HIDDEN_SIZE = int(os.getenv('OCR_HIDDEN_SIZE', '64'))
CONF_THRESHOLD = float(os.getenv('OCR_CONF_THRESHOLD', '0.25'))
AREA_CONF_THRESHOLD = float(os.getenv('OCR_AREA_CONF_THRESHOLD', str(CONF_THRESHOLD)))

CRNN_NAING_TOKEN = '\u1014\u102d\u102f\u1004\u103a'
CRNN_NAING_PLACEHOLDER = '\ue000'

DEFAULT_BURMESE_DIGITS = ['\u1040', '\u1041', '\u1042', '\u1043', '\u1044', '\u1045', '\u1046', '\u1047', '\u1048', '\u1049']
DIGIT_NAME_MAP = {f'digit_{i}': DEFAULT_BURMESE_DIGITS[i] for i in range(10)}
DEFAULT_CRNN_CHARSET = (
    DEFAULT_BURMESE_DIGITS
    + [CRNN_NAING_PLACEHOLDER]
    + [
        '\u1000', '\u1001', '\u1002', '\u1003', '\u1004', '\u1005', '\u1006', '\u1007',
        '\u100a', '\u100f', '\u1010', '\u1011', '\u1012', '\u1013', '\u1014', '\u1015',
        '\u1016', '\u1017', '\u1018', '\u1019', '\u101a', '\u101b', '\u101c', '\u101d',
        '\u101e', '\u101f', '\u1021', '\u1025'
    ]
)
CRNN_CLASSES_PATH = Path(
    os.getenv('OCR_CRNN_CLASSES_PATH', MODELS_DIR / 'charset_39.txt')
).resolve()


def _load_crnn_classes(path):
    try:
        if path.exists():
            lines = [line.strip().lstrip('\ufeff') for line in path.read_text(encoding='utf-8').splitlines()]
            lines = [line for line in lines if line]
            if lines:
                normalized = []
                for name in lines:
                    normalized.append(DIGIT_NAME_MAP.get(name, name))
                return normalized
    except Exception:
        return None
    return None


BURMESE_DIGITS = _load_crnn_classes(CRNN_CLASSES_PATH) or DEFAULT_CRNN_CHARSET
BURMESE_TO_LATIN = str.maketrans('\u1040\u1041\u1042\u1043\u1044\u1045\u1046\u1047\u1048\u1049', '0123456789')
