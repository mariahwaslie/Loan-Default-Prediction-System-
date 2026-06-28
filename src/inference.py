import os
import time
from pathlib import Path

import joblib
import pandas as pd

from src.logging_config import get_logger
from src.preprocessing import prepare_input


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "loan_default_model.pkl"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

logger = get_logger(__name__)

# --------------------------------------------------
# LOAD MODEL ONCE (IMPORTANT OPTIMIZATION)
# --------------------------------------------------

def load_model():
    """
    Load model artifact once per process.
    """
    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["model"],
        artifact["features"],
        artifact["cat_features"],
        artifact["threshold"],
    )


MODEL, FEATURES, CAT_FEATURES, THRESHOLD = load_model()


# --------------------------------------------------
# CORE PREDICTION LOGIC
# --------------------------------------------------

def _predict_proba(df: pd.DataFrame) -> float:
    """
    Internal probability prediction.
    """
    return MODEL.predict_proba(df)[0][1]


# --------------------------------------------------
# PUBLIC FUNCTION (USED BY FASTAPI)
# --------------------------------------------------

def predict(data: dict) -> dict:
    """
    Full inference pipeline:
    - input validation
    - preprocessing
    - prediction
    - thresholding
    - logging
    """

    start_time = time.perf_counter()

    try:
        # 1. prepare input (fills defaults + aligns features)
        df = prepare_input(
            data,
            FEATURES,
            CAT_FEATURES
        )

        # 2. probability prediction
        probability = _predict_proba(df)

        # 3. classification
        prediction = int(probability >= THRESHOLD)

        # 4. timing
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # 5. logging success
        logger.info(
            "event=prediction_completed inference_time_ms=%.2f probability=%.6f prediction=%s",
            inference_time_ms,
            probability,
            prediction,
        )

        return {
            "default_probability": float(probability),
            "prediction": prediction
        }

    except Exception as e:
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "event=prediction_failed inference_time_ms=%.2f error=%s",
            inference_time_ms,
            str(e),
        )
        raise