from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute regression metrics for a model on labeled data.

    What:
    - Evaluates a trained model on a feature matrix `X` and true labels `y`.

    Why:
    - Measurement is required to understand model quality and compare
    model variants before saving or deploying.

    How:
    - Uses the model's `predict` method to generate predictions.
    - Computes a set of regression metrics from predictions and true labels.
    """
    preds = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "r2": float(r2_score(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "median_ae": float(median_absolute_error(y, preds)),
        "explained_variance": float(explained_variance_score(y, preds)),
        "max_error": float(max_error(y, preds)),
    }
