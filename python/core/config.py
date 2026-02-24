from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv('OCR_MODELS_DIR', ROOT_DIR / 'models')).resolve()

YOLO_WEIGHTS = Path(os.getenv('OCR_YOLO_WEIGHTS', MODELS_DIR / 'best.pt')).resolve()
AREA_YOLO_WEIGHTS = Path(os.getenv('OCR_AREA_YOLO_WEIGHTS', MODELS_DIR / 'best_area.pt')).resolve()
CRNN_WEIGHTS = Path(os.getenv('OCR_CRNN_WEIGHTS', MODELS_DIR / 'crnn_burmese.pth')).resolve()

IMG_HEIGHT = int(os.getenv('OCR_IMG_HEIGHT', '32'))
IMG_WIDTH = int(os.getenv('OCR_IMG_WIDTH', '128'))
HIDDEN_SIZE = int(os.getenv('OCR_HIDDEN_SIZE', '128'))
CONF_THRESHOLD = float(os.getenv('OCR_CONF_THRESHOLD', '0.25'))
AREA_CONF_THRESHOLD = float(os.getenv('OCR_AREA_CONF_THRESHOLD', str(CONF_THRESHOLD)))

BURMESE_DIGITS = ['၀', '၁', '၂', '၃', '၄', '၅', '၆', '၇', '၈', '၉']
BURMESE_TO_LATIN = str.maketrans('၀၁၂၃၄၅၆၇၈၉', '0123456789')
