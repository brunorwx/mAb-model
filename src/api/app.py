from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.api.services.prediction import load_predictor

# This module initializes the FastAPI app and wires the application
# lifecycle so the predictor artifact is loaded once at startup.
# Routing is delegated to `src.api.routes` and artifact loading is delegated
# to `src.api.services.prediction`.


@asynccontextmanager
async def lifespan(app: FastAPI):
    # What: load the saved model and preprocessing artifacts during startup.
    # Why: keep inference fast and deterministic by loading artifacts once,
    # instead of on every request.
    # How: call `load_predictor()` and store the result on `app.state.predictor`.
    app.state.predictor = load_predictor()
    yield


app = FastAPI(
    title="mAb Titer Prediction API",
    version="0.1.0",
    description="Minimal inference API for the mAb prediction pipeline.",
    lifespan=lifespan,
)
app.include_router(router)
