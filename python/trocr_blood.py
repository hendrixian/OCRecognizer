from pathlib import Path
import os
import threading
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torch.nn import functional as F
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv('OCR_MODELS_DIR', ROOT_DIR / 'models')).resolve()
MODEL_PATH = Path(
    os.getenv('OCR_TROCR_BLOOD_MODEL', MODELS_DIR / 'myanmar_bloodtype_thresholded')
).resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_processor = None
_model = None
_load_lock = threading.Lock()


def _ensure_loaded():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    with _load_lock:
        if _processor is None or _model is None:
            processor = TrOCRProcessor.from_pretrained(
                str(MODEL_PATH),
                local_files_only=True,
                use_fast=False
            )
            model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH), local_files_only=True)
            model.to(device)
            model.eval()
            _processor = processor
            _model = model

    return _processor, _model


@torch.no_grad()
def read_blood_type(pil_image: Image.Image):
    processor, model = _ensure_loaded()
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

    outputs = model.generate(
        pixel_values,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=3   # blood type is short → makes it faster
    )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    # confidence
    scores = outputs.scores
    probs = [F.softmax(s, dim=-1).max().item() for s in scores]
    confidence = sum(probs) / len(probs) if probs else 1.0

    return text.strip(), confidence
