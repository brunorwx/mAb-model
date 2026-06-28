from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from src.preprocessing.pipeline import load_preprocessor_artifact

# Small inference helper layer for artifact loading and prediction logic.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PACKAGE_ROOT / "src" / "artifacts"
ARTIFACT_MODEL_FILE = ARTIFACT_DIR / "xgb_model.pkl"
ARTIFACT_PREPROCESSOR_FILE = ARTIFACT_DIR / "scaler_preprocessor.pkl"


class Predictor:
    def __init__(
        self,
        model: object,
        scaler: object,
        feature_names: list[str],
        name_mapping: dict[str, str],
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.name_mapping = name_mapping


def load_predictor() -> Predictor:
    """Load the production model and preprocessor artifact.

    Reads the saved model and preprocessor files from disk and returns a
    `Predictor` instance.
    """
    if not ARTIFACT_PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {ARTIFACT_PREPROCESSOR_FILE}. Build the container with preprocessing artifacts baked in."
        )
    if not ARTIFACT_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_MODEL_FILE}. Train and save the model before serving."
        )

    scaler, feature_names, name_mapping = load_preprocessor_artifact(
        path=ARTIFACT_PREPROCESSOR_FILE
    )

    with ARTIFACT_MODEL_FILE.open("rb") as handle:
        model = pickle.load(handle)

    return Predictor(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        name_mapping=name_mapping,
    )


def preprocess_features(
    features: dict[str, float], predictor: Predictor
) -> pd.DataFrame:
    """Validate and shape raw request features into the model input DataFrame."""
    mapped_features: dict[str, float] = {}
    unknown: list[str] = []

    for raw_name, value in features.items():
        if raw_name in predictor.feature_names:
            mapped_features[raw_name] = value
        elif raw_name in predictor.name_mapping:
            mapped_features[predictor.name_mapping[raw_name]] = value
        else:
            unknown.append(raw_name)

    if unknown:
        raise ValueError(
            f"Unexpected feature names: {unknown}. Expected sanitized names or original training names."
        )

    missing = [name for name in predictor.feature_names if name not in mapped_features]
    if missing:
        raise ValueError(
            f"Missing features: {missing}. Expected feature names: {predictor.feature_names}"
        )

    X = pd.DataFrame([mapped_features])
    X = X[predictor.feature_names]
    return X


def predict_titer(predictor: Predictor, features: dict[str, float]) -> float:
    """Run a prediction using the preloaded model and preprocessor."""
    if not features:
        raise ValueError("Request must include feature values.")

    X = preprocess_features(features, predictor)
    X_scaled = predictor.scaler.transform(X.values)
    preds = predictor.model.predict(X_scaled)
    return float(np.squeeze(preds))
