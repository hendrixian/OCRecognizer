import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from core.config import BLOOD_TYPE_MODEL_DIR

app = FastAPI(title='NRC OCR Service', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*']
)
app.include_router(router)

if BLOOD_TYPE_MODEL_DIR.exists():
    app.mount('/blood-type-model', StaticFiles(directory=str(BLOOD_TYPE_MODEL_DIR)), name='blood_type_model')


@app.get('/blood-type-model-manifest')
def blood_type_model_manifest():
    if not BLOOD_TYPE_MODEL_DIR.exists():
        return {'available': False, 'modelDirs': [], 'classes': []}

    model_dirs = sorted(
        entry.name
        for entry in BLOOD_TYPE_MODEL_DIR.iterdir()
        if entry.is_dir() and (entry / 'model.json').exists()
    )

    classes = []
    info_path = BLOOD_TYPE_MODEL_DIR / 'ensemble_info.json'
    if info_path.exists():
        try:
            parsed = json.loads(info_path.read_text(encoding='utf-8'))
            parsed_classes = parsed.get('classes')
            if isinstance(parsed_classes, list):
                classes = [str(value) for value in parsed_classes]
        except Exception:
            classes = []

    return {
        'available': bool(model_dirs),
        'modelDirs': model_dirs,
        'classes': classes
    }
