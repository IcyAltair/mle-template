from pydantic import BaseModel

class PredictResponse(BaseModel):
    dogProbability: float