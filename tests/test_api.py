import pickle
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.api.app import app
from src.api.services import prediction as prediction_module


def _build_test_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    feature_names = ["a", "b"]
    name_mapping = {"a": "a", "b": "b"}

    scaler = StandardScaler()
    scaler.fit(np.array([[1.0, 2.0], [2.0, 3.0]]))

    preprocessor_path = tmp_path / "preprocessor.pkl"
    with preprocessor_path.open("wb") as handle:
        pickle.dump(
            {
                "scaler": scaler,
                "feature_names": feature_names,
                "name_mapping": name_mapping,
            },
            handle,
        )

    model = LinearRegression()
    model.fit(np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([1.0, 2.0]))
    model_path = tmp_path / "xgb_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    return preprocessor_path, model_path


def test_api_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_predict_endpoint(tmp_path: Path):
    preprocessor_path, model_path = _build_test_artifacts(tmp_path)

    prediction_module.set_artifact_paths(preprocessor_path, model_path)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "exp": 1,
                "features": {"a": 1.0, "b": 2.0},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["exp"] == 1
    assert payload["details"] == "prediction successful"
    assert isinstance(payload["titer"], float)
