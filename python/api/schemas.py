from typing import List, Optional
from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    image_base64: str = Field(..., description='Image as base64 or data URL')
    conf_threshold: Optional[float] = Field(0.25, ge=0.0, le=1.0)


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: float


class RegionBox(Box):
    label: str


class ScanResponse(BaseModel):
    nrcNumber: str
    nrcNumberBurmese: str
    rawDigits: str
    confidence: float
    bloodType: Optional[str] = ''
    bloodTypeConfidence: float = 0.0
    bloodTypeBox: Optional[RegionBox] = None
    boxes: List[Box]
    regionBoxes: List[RegionBox] = []
    inferenceMs: float
    model: dict
