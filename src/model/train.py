from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.model.evaluate import evaluate_model
from src.preprocessing.pipeline import build_train_pipeline

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PACKAGE_ROOT / "src" / "artifacts"
XGB_MODEL_PATH = ARTIFACT_DIR / "xgb_model.pkl"
LGBM_MODEL_PATH = ARTIFACT_DIR / "lgbm_model.pkl"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> XGBRegressor:
    """Train and return an XGBoost regression model.

    What:
    - Constructs an XGBoost regressor and fits it to the provided feature
      matrix `X` and target vector `y`.

    Why:
    - XGBoost is a high-performance gradient boosting implementation that
      often delivers strong regression accuracy on tabular data.

    How:
    - Creates the model with deterministic settings and parallel training,
      then calls `fit(X, y)` to learn the tree ensemble.
    """

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=-1,
        verbosity=1,
    )
    model.fit(X, y)
    return model


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> LGBMRegressor:
    """Train and return a LightGBM regression model.

    What:
    - Builds a LightGBM regressor and fits it on the training data.

    Why:
    - LightGBM is another gradient boosting algorithm that can provide
      fast training and good generalization for tabular features.

    How:
    - Instantiates the model with the specified hyperparameters and then
      trains it by calling `fit(X, y)`.
    """

    model = LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, y)
    return model


def save_model(model: object, path: Path) -> None:
    """Save a trained model object to disk using pickle.

    What:
    - Persists a fitted model object to a file so it can be loaded later for
      inference or evaluation.

    Why:
    - Saving the model avoids retraining and preserves the exact learned
      parameters for production deployment.

    How:
    - Ensures the target directory exists and writes the model object using
      `pickle.dump`.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def train_all_models() -> dict[str, dict[str, float]]:
    """Train XGBoost and LightGBM on the preprocessed training data and save artifacts.

    What:
    - Runs the full training workflow: preprocess data, fit both models, save
      them, and return training metrics.

    Why:
    - Encapsulates the end-to-end model training flow in one callable function
      for repeatable experimentation and easy execution.

    How:
    - Loads preprocessed data from `build_train_pipeline()`.
    - Trains XGBoost and LightGBM models on the same dataset.
    - Saves the resulting model files to the artifacts directory.
    - Computes and returns RMSE/R^2 for each trained model.
    """

    X_train, y_train, _ = build_train_pipeline()

    xgb_model = train_xgboost(X_train, y_train)
    lgbm_model = train_lightgbm(X_train, y_train)

    save_model(xgb_model, XGB_MODEL_PATH)
    save_model(lgbm_model, LGBM_MODEL_PATH)

    return {
        "xgboost": evaluate_model(xgb_model, X_train, y_train),
        "lightgbm": evaluate_model(lgbm_model, X_train, y_train),
    }


if __name__ == "__main__":
    metrics = train_all_models()
    print("Saved model artifacts:")
    print(f"- {XGB_MODEL_PATH}")
    print(f"- {LGBM_MODEL_PATH}")
    print("Training metrics:")
    for model_name, model_metrics in metrics.items():
        print(
            f"{model_name}: rmse={model_metrics['rmse']:.4f}, r2={model_metrics['r2']:.4f}"
        )
