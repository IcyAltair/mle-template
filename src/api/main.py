from fastapi import FastAPI
from .routes import router
from src.models.CatVDogModel import CatVDogModel

app = FastAPI(title="Dog vs Cat API")
service = CatVDogModel(config_path="config.ini", show_log=True)
service.set_device("cpu")
service.load_classifier("LOG_REG")

app.include_router(router)