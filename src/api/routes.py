from fastapi import APIRouter, HTTPException, Request, status

from src.api.dto.requests import PredictRequest
from src.api.dto.responses import HealthResponse, PredictResponse
from src.api.services.prediction import predict_titer

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    What: returns a simple OK payload indicating the service is running.
    Why: useful for liveness/readiness probes and quick sanity checks.
    How: returns a `HealthResponse` object with a fixed `status`.
    """
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, http_request: Request):
    """Prediction endpoint for titer inference.

    What: accepts experiment metadata and numeric features, then returns a
    predicted titer value.
    Why: exposes the pre-trained model through a simple REST API for
    downstream clients or integration tests.
    How: retrieves the preloaded predictor from app state, validates the
    request schema, invokes `predict_titer`, and wraps the result in a
    response model.
    """
    predictor = getattr(http_request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not available. Check server logs.",
        )

    try:
        titer = predict_titer(predictor, request.features)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return PredictResponse(
        exp=request.exp, titer=titer, details="prediction successful"
    )
