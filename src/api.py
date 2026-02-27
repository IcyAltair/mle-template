from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/model.joblib")

CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


class PredictRequest(BaseModel):
    pixels: Optional[List[float]] = Field(default=None, description="Length 784, values in [0..1] or [0..255]")
    fill: Optional[float] = Field(default=None, description="Fill all 784 pixels with this value")
    random_seed: Optional[int] = Field(default=None, description="Generate deterministic random pixels")


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    proba: List[float]


app = FastAPI(title="Fashion-MNIST Classic ML API")
_model = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run training or dvc pull artifacts."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    ok = MODEL_PATH.exists()
    return {"status": "ok", "model_present": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _load_model()

    if req.pixels is not None:
        if len(req.pixels) != 784:
            raise HTTPException(status_code=400, detail="pixels must have length 784")
        x = np.array(req.pixels, dtype=np.float32)
    elif req.fill is not None:
        x = np.full((784,), float(req.fill), dtype=np.float32)
    elif req.random_seed is not None:
        rng = np.random.default_rng(int(req.random_seed))
        x = rng.random(784, dtype=np.float32)
    else:
        raise HTTPException(status_code=400, detail="Provide pixels OR fill OR random_seed")

    if x.max() > 1.5:
        x = x / 255.0

    X = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].tolist()
        class_id = int(np.argmax(proba))
    else:
        class_id = int(model.predict(X)[0])
        proba = [0.0] * 10
        proba[class_id] = 1.0

    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES.get(class_id, str(class_id)),
        "proba": [float(p) for p in proba],
    }