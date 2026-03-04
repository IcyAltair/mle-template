from fastapi import APIRouter, File, UploadFile, HTTPException
from .schemas import PredictResponse
import numpy as np

router = APIRouter()

def get_service():
    from .main import service
    return service

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    data = await file.read()
    svc = get_service()

    from io import BytesIO
    from PIL import Image

    img = Image.open(BytesIO(data)).convert("RGB")
    emb = svc.embed_pil(img).reshape(1, -1)

    clf = svc.classifier
    if clf is None:
        raise HTTPException(status_code=500, detail="Classifier is not loaded")

    if not hasattr(clf, "predict_proba"):
        raise HTTPException(status_code=500, detail="Classifier does not support predict_proba")

    proba = clf.predict_proba(emb)

    dog_class_index = 1
    dog_prob = float(proba[0, dog_class_index])

    return PredictResponse(dogProbability=dog_prob)