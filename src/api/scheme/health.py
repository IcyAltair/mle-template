from pydantic import BaseModel

import time
from typing import List



class SiteHealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float

class ModelsHealthResponse(BaseModel):
    status: str
    models_loaded: int
    models_available: List[str]
    models_missing: List[str]
