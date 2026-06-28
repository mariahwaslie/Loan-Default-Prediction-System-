from fastapi import APIRouter
from typing import Dict, Any
import time

from src.inference import predict
from src.logging_config import get_logger
from src.schemas import LoanRequest
router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
def root():
    return {"status": "running", "message": "Loan Default API is live"}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")

def predict_loan(data: LoanRequest):

    """
    Predict loan default probability using full validated schema.
    """

    start_time = time.perf_counter()

    logger.info(
        "event=prediction_request_received loan_amnt=%s dti=%s",
        data.loan_amnt,
        data.dti,
    )

    try:

        # convert Pydantic → dict
        payload = data.model_dump()

        # optional debug log
        logger.debug("raw_input=%s", payload)

        # inference
        result = predict(payload)

        logger.info(

            "event=prediction_success probability=%.6f prediction=%s",
            result["default_probability"],
            result["prediction"],
        )

        return result
    except Exception as e:

        logger.exception(
            "event=prediction_failed error=%s",
            str(e),

        )

        raise

    finally:

        duration = time.perf_counter() - start_time
        logger.info(
            "event=prediction_latency seconds=%.4f",
            duration,

        )