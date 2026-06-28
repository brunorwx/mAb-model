import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from src.model.train import (
    LGBMRegressor,
    XGBRegressor,
    evaluate_model,
    save_model,
    train_lightgbm,
    train_xgboost,
)


def test_train_xgboost_basic():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    y = pd.Series([0.5, 1.0, 1.5])

    model = train_xgboost(X, y, n_estimators=10, max_depth=2)

    assert isinstance(model, XGBRegressor)
    preds = model.predict(X)
    assert preds.shape == (3,)
    assert np.allclose(preds, [0.5, 1.0, 1.5], atol=0.5)


def test_train_lightgbm_basic():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    y = pd.Series([0.5, 1.0, 1.5])

    model = train_lightgbm(X, y, n_estimators=10, max_depth=2)

    assert isinstance(model, LGBMRegressor)
    preds = model.predict(X)
    assert preds.shape == (3,)
    assert np.allclose(preds, [0.5, 1.0, 1.5], atol=0.5)


def test_save_model_and_load(tmp_path: Path):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = train_xgboost(X, y, n_estimators=5, max_depth=2)

    path = tmp_path / "model.pkl"
    save_model(model, path)

    with path.open("rb") as handle:
        loaded = pickle.load(handle)

    assert hasattr(loaded, "predict")
    assert np.allclose(loaded.predict(X), model.predict(X))


def test_evaluate_model_outputs_metrics():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, 2.0])
    model = train_lightgbm(X, y, n_estimators=5, max_depth=2)

    metrics = evaluate_model(model, X, y)
    assert set(metrics) == {
        "rmse",
        "r2",
        "mae",
        "median_ae",
        "explained_variance",
        "max_error",
    }
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["median_ae"] >= 0
    assert metrics["max_error"] >= 0
    assert -1.0 <= metrics["r2"] <= 1.0
    assert -1.0 <= metrics["explained_variance"] <= 1.0
