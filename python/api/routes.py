from fastapi import APIRouter, Body, HTTPException
from api.schemas import ScanRequest, ScanResponse
from core.recognizer import get_recognizer
from utils.image import decode_base64_image

router = APIRouter()


@router.get('/health')
def health():
    return {'status': 'ok'}


@router.post('/scan', response_model=ScanResponse)
def scan(payload: ScanRequest = Body(...)):
    try:
        image = decode_base64_image(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recognizer = get_recognizer()
    result = recognizer.recognize(image, conf_threshold=payload.conf_threshold)
    return result
