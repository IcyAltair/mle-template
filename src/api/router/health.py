from fastapi import APIRouter

from src.api.scheme.health import SiteHealthResponse
from src.api.scheme.health import ModelsHealthResponse
import time

router = APIRouter()

start_time = time.time()

@router.get("/health", response_model=SiteHealthResponse)
async def site_health_check() -> SiteHealthResponse:
    uptime = time.time() - start_time
    return SiteHealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(uptime, 2)
    )

classifiers = {}

@router.get("/models/health", response_model=ModelsHealthResponse)
async def models_health_check() -> ModelsHealthResponse:
    expected_models = ["LOG_REG", "RAND_FOREST", "KNN", "SVM", "GNB", "D_TREE"]
    loaded = list(classifiers.keys())
    missing = [m for m in expected_models if m not in loaded]
    status = "healthy" if not missing else "degraded"
    return ModelsHealthResponse(
        status=status,
        models_loaded=len(loaded),
        models_available=loaded,
        models_missing=missing
    )