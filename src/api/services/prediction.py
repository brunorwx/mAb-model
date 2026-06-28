"""Thin API-facing service that delegates inference to the model layer.

This module intentionally avoids re-exporting artifact paths or model types.
To customize artifact locations in tests or runtime, call `set_artifact_paths()`
before `load_predictor()` is invoked by the app lifecycle.
"""

from pathlib import Path
from src.model import inference as _inference


def set_artifact_paths(preprocessor_path: Path, model_path: Path) -> None:
    """Set artifact file paths on the model inference module.

    Tests can call this before the app startup to point the inference layer at
    temporary artifact files.
    """
    _inference.ARTIFACT_PREPROCESSOR_FILE = Path(preprocessor_path)
    _inference.ARTIFACT_MODEL_FILE = Path(model_path)


def load_predictor():
    return _inference.load_predictor()


def preprocess_features(features: dict[str, float], predictor):
    return _inference.preprocess_features(features, predictor)


def predict_titer(predictor, features: dict[str, float]) -> float:
    return _inference.predict_titer(predictor, features)
