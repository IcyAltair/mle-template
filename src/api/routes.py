from fastapi import APIRouter, File, UploadFile, HTTPException
from .schemas import PredictResponse

router = APIRouter()

def get_service():
    from main import service
    return service

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    data = await file.read()
    svc = get_service()
    p = svc.predict_dog_proba_bytes(data)
    return PredictResponse(dogProbability=p)