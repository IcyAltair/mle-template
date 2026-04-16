import uvicorn
from fastapi import FastAPI, HTTPException, Query

from src.api.router.health import router as health_router
from src.api.router.model import router as model_router



app = FastAPI(title="Iris Classifier API", version="1.0.0")

app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(model_router, prefix="/model", tags=["model"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)