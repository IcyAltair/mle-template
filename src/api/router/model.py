from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import configparser
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import traceback

from src.logger import Logger
from src.preprocess import DataMaker
from src.train import MultiModel

SHOW_LOG = True

router = APIRouter()

logger_instance = Logger(SHOW_LOG)
log = logger_instance.get_logger(__name__)

config = configparser.ConfigParser()

classifiers = {}
sc = None


class IrisFeatures(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float




class PredictionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float


def initialize_models():
    global config, classifiers, sc

    try:
        
        config.read("config.ini")

        X_train = pd.read_csv(config["SPLIT_DATA"]["X_train"], index_col=0)
        # X_test = pd.read_csv(config["SPLIT_DATA"]["X_test"], index_col=0)

        sc = StandardScaler()
        sc.fit(X_train)

        # models = ["LOG_REG", "RAND_FOREST", "KNN", "SVM", "GNB", "D_TREE"]
        models = ["LOG_REG"]
        for model_name in models:
            model_path = config[model_name]["path"]
            if os.path.isfile(model_path):
                with open(model_path, "rb") as f:
                    classifiers[model_name] = pickle.load(f)
                log.info(f"Loaded {model_name}")
            else:
                log.warning(f"Model {model_name} not found at {model_path}")

        log.info(f"Initialized {len(classifiers)} models")
    except Exception as e:
        log.error(traceback.format_exc())
        sys.exit(1)


@router.on_event("startup")
async def startup_event():
    initialize_models()





@router.post("/predict", response_model=PredictionResponse)
async def predict(
    features: IrisFeatures,
    model: str = Query("LOG_REG", description="Model name")
):
    if model not in classifiers:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model} not available. Available models: {list(classifiers.keys())}"
        )

    try:
        X = pd.DataFrame([[
            features.feature_0,
            features.feature_1,
            features.feature_2,
            features.feature_3
        ]])

        X_scaled = sc.transform(X)

        classifier = classifiers[model]
        prediction = classifier.predict(X_scaled)[0]

        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(X_scaled)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 1.0

        log.info(f"Prediction with {model}: {prediction}")

        return PredictionResponse(
            model=model,
            prediction=prediction,
            confidence=confidence
        )

    except Exception as e:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

