"""Preprocessing pipeline implementing README steps with rationale and details.

This module centralizes dataset loading, transformation, and scaling used by the
training and inference workflows. The dataset contains variable-length
time-series rows per experiment (`Exp`) and a small set of Z:* (setpoint)
columns that are constant per experiment (generally measured at day 0). The
pipeline converts the row-level time-series into a single fixed-length row
per experiment suitable for classical ML models.

High-level steps and rationale
- Load CSVs from the `data/` folder (`load_data`).
- Forward-fill `Z:*` columns per experiment (`forward_fill_z_cols`) so that
    experiments which only report these setpoints at the first row retain them
    across the group's rows. Forward-filling within groups avoids leaking
    information between experiments.
- Aggregate variable-length time-series into fixed-length features
    (`aggregate_experiment_features`):
    - For `Z:*` columns we take the first value per experiment (after
        forward-fill they are effectively constant) because they represent
        initial setpoints or static metadata.
    - For other numeric columns we compute summary statistics: last value,
        mean and standard deviation. Rationale:
            - `last` captures the most recent measurement (often most predictive
                for end outcomes like titer).
            - `mean` captures the central tendency across the experiment.
            - `std` captures variability which can be informative for process
                stability or noise.
    - We drop `RowID` and `Time[day]` from features because they are row-level
        identifiers/timestamps rather than predictive features once aggregated.
- Fit a `StandardScaler` on the training features only (`fit_scaler`) and
    apply it to train and test using `apply_scaler`. Fitting only on training
    data prevents data leakage from validation/test sets.
- Save processed CSVs and the fitted scaler artifact to `data/artifacts`
    when requested. Persisting the scaler allows consistent preprocessing in
    deployment.

Design notes and data-leakage considerations
- The pipeline sets the experiment id (`Exp`) as the index before joining
    train targets. This ensures the features align correctly with targets.
- The scaler is fit on the training matrix `X` only. The same scaler object
    must be applied to test/unseen data. Avoid re-fitting the scaler on
    test/validation data to prevent optimistic performance estimates.

Functions expose small, testable units (load, forward-fill, aggregate,
scaling, build pipelines) to keep behavior clear and make unit testing
straightforward.

What and Why (concise)
- What is being produced: the pipeline converts row-level time-series data
    (multiple measurements per `Exp`) into a single fixed-length feature row
    per experiment. The outputs are:
        - a feature matrix `X` (one row per `Exp`) and
        - a target vector `y` (the per-experiment `Y:Titer`).
- What is a scaler: a scaler (here `StandardScaler`) is an object that
    records per-feature mean and variance from training data and can transform
    raw features into z-scores: (x - mean) / std. This centers features to
    zero mean and scales them to unit variance.
- Why use a scaler: many ML models assume or benefit from features on a
    comparable scale (e.g., linear models, gradient-based optimizers). Scaling
    prevents features with large numeric ranges from dominating the model and
    can improve numerical stability and convergence.

Implementation note: we intentionally fit the scaler only on training data
and persist the fitted scaler so that inference uses the exact same
transformation.
"""

from pathlib import Path
import pickle
import re
from collections import Counter

import pandas as pd
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data"
ARTIFACT_DIR = PACKAGE_ROOT / "src" / "artifacts"
PREPROCESSOR_ARTIFACT_PATH = ARTIFACT_DIR / "scaler_preprocessor.pkl"


def load_data(data_dir: Path = DATA_DIR):
    """Load CSVs from `data_dir` and return (train_data, train_targets, test_data).

    Expects the following files to exist under `data_dir`:
    - `datahow_interview_train_data.csv`: row-level time-series with `Exp`
        grouping and `RowID`/`Time[day]` columns.
    - `datahow_interview_train_targets.csv`: per-experiment targets with an
        `Exp` column and `Y:Titer` target.
    - `datahow_interview_test_data.csv`: row-level time-series for test
        experiments (no targets).

    Returns pandas DataFrames in the order (train_data, train_targets, test_data).
    """

    train_data = pd.read_csv(data_dir / "datahow_interview_train_data.csv")
    train_targets = pd.read_csv(data_dir / "datahow_interview_train_targets.csv")
    test_data = pd.read_csv(data_dir / "datahow_interview_test_data.csv")

    return train_data, train_targets, test_data


def forward_fill_z_cols(df: pd.DataFrame):
    """Forward-fill Z:* columns within each `Exp` group.

    Some experiments only record setpoints (`Z:*` columns) on the first
    (day 0) row. Forward-filling them within each `Exp` group ensures that
    aggregation (which looks at per-column summaries) sees the setpoint for
    every row without copying values between experiments.

    Operates in-place on a column slice but returns the full DataFrame for
    convenience. If no `Z:` columns exist the original DataFrame is
    returned unchanged.
    """

    z_cols = [c for c in df.columns if c.startswith("Z:")]

    if not z_cols:
        return df
    df[z_cols] = df.groupby("Exp")[z_cols].ffill()

    return df


def sanitize_feature_names(
    columns: pd.Index,
) -> tuple[pd.Index, dict[str, str]]:
    """Convert feature names to LightGBM-safe names and preserve a mapping."""

    sanitized = []
    mapping: dict[str, str] = {}
    counts = Counter()
    for col in columns:
        safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", col)
        safe_name = re.sub(r"__+", "_", safe_name).strip("_")
        if not safe_name:
            safe_name = "feature"
        counts[safe_name] += 1
        if counts[safe_name] > 1:
            safe_name = f"{safe_name}_{counts[safe_name] - 1}"
        mapping[col] = safe_name
        sanitized.append(safe_name)

    return pd.Index(sanitized), mapping


def save_preprocessor_artifact(
    scaler: StandardScaler,
    feature_names: list[str],
    mapping: dict[str, str],
    path: Path = PREPROCESSOR_ARTIFACT_PATH,
) -> None:
    """Persist preprocessing metadata for production inference.

    What: save the fitted scaler, feature order, and sanitized feature name
    mapping into a single pickle artifact.
    Why: inference must apply the exact same scaling and feature ordering used
    during training, otherwise predictions may be inconsistent.
    How: create the artifact directory if needed and serialize a dictionary
    containing the scaler and metadata.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {"scaler": scaler, "feature_names": feature_names, "name_mapping": mapping},
            handle,
        )


def load_preprocessor_artifact(
    path: Path = PREPROCESSOR_ARTIFACT_PATH,
) -> tuple[StandardScaler, list[str], dict[str, str]]:
    """Load saved preprocessing metadata for inference.

    What: read the saved artifact produced during training.
    Why: the API needs the exact same scaler, feature order, and mapping used
    by the training pipeline to transform incoming requests correctly.
    How: deserialize the pickle file and return the stored objects.
    """

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    return payload["scaler"], payload["feature_names"], payload["name_mapping"]


def aggregate_experiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level timeseries DataFrame into one row per `Exp`.

    Inputs
    - `df`: row-level DataFrame containing `Exp` column and various numeric
        columns. `RowID` and `Time[day]` are treated as non-feature columns
        and removed from aggregation.

    Aggregation strategy and why
    - `Z:*` columns: these represent static setpoints or metadata. After
        forward-filling, they are constant for an experiment so we take the
        first value to represent the experiment-level attribute.
    - Other numeric columns: we compute three summaries per column:
            - `_last`: the final observed value for the experiment (captures
                the end-state behavior which often correlates with outcomes).
            - `_mean`: average over the experiment (captures central tendency).
            - `_std`: variability (noise or stability of the process). Missing
                std values (single-row experiments) are filled with 0.

    Returns a DataFrame indexed by `Exp` (then reset to make `Exp` a column)
    where each original column has been converted into 1 or 3 feature
    columns depending on whether it was a `Z:` column or not.
    """

    df = df.copy()

    drop_cols = [c for c in ("RowID", "Time[day]") if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["Exp"]]

    z_cols = [c for c in feature_cols if c.startswith("Z:")]
    other_cols = [c for c in feature_cols if c not in z_cols]

    groups = df.groupby("Exp")
    out = pd.DataFrame(index=groups.groups.keys())

    # Z columns: take first (after forward-fill they are constant)
    for c in z_cols:
        out[c] = groups[c].first()

    # columns: last / mean / std
    for c in other_cols:
        out[f"{c}_last"] = groups[c].last()
        out[f"{c}_mean"] = groups[c].mean()
        out[f"{c}_std"] = groups[c].std().fillna(0)

    out.index.name = "Exp"
    out = out.reset_index()
    return out


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fit a sklearn `StandardScaler` on the training feature matrix.

    What (short):
    - Creates a `StandardScaler` and fits it to `X` so it learns per-feature
        mean and standard deviation.

    Why (short):
    - Standardization (subtract mean, divide by std) produces features with
        zero mean and unit variance which many models prefer. It also prevents
        high-variance features from dominating distance-based or gradient-based
        learning.

    Implementation details:
    - We pass `X.values` to `scaler.fit()` to ensure the scaler only sees
        numeric arrays (no index or pandas metadata). The fitted scaler stores
        `scale_` and `mean_` attributes which are saved as the preprocessing
        artifact and later reused during inference.
    - Fit only on training features to avoid data leakage into evaluation.
    """

    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a fitted `scaler` to DataFrame `X` and return a new DataFrame.

    What (short):
    - Uses the provided `scaler` to transform raw features into standardized
        values using the scaler's stored mean and variance.

    Why (short):
    - Applying the exact same scaler used in training to any new data
        (validation/test/inference) ensures consistent feature distributions
        and prevents mismatch between training and inference preprocessing.

    Notes:
    - The function preserves column names and index so the downstream code
        (models, logging, diagnostics) can reference features by name.
    - If you ever need to reverse the scaling (e.g., to interpret outputs
        that were transformed), use `scaler.inverse_transform` on the scaled
        array and re-create a DataFrame with the original column names.
    """

    arr = scaler.transform(X.values)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def build_train_pipeline() -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Build the full training preprocessing pipeline.

    Steps performed:
    1. Load raw train data and targets.
    2. Forward-fill `Z:*` columns so setpoints are available for aggregation.
    3. Aggregate row-level timeseries into one row per `Exp`.
    4. Align targets by setting `Exp` as index and joining `Y:Titer`.
    5. Fit a `StandardScaler` on the training features and transform them.
    6. Optionally save processed CSVs and the fitted scaler to `out_dir`.

    Returns:
    - `X_scaled`: scaled feature DataFrame indexed by `Exp`.
    - `y`: Series of training targets aligned with `X_scaled`.
    - `scaler`: the fitted `StandardScaler` instance (use for test data).
    """

    train_data, train_targets, _ = load_data()
    train_data = forward_fill_z_cols(train_data)
    X = aggregate_experiment_features(train_data)

    feature_cols = [c for c in X.columns if c != "Exp"]
    _, mapping = sanitize_feature_names(pd.Index(feature_cols))
    X = X.rename(columns=mapping)

    y = train_targets.set_index("Exp")["Y:Titer"]
    X = X.set_index("Exp")
    X = X.join(y, how="left")
    y = X.pop("Y:Titer")

    scaler = fit_scaler(X)
    X_scaled = apply_scaler(X, scaler)

    save_preprocessor_artifact(
        scaler=scaler, feature_names=list(X.columns), mapping=mapping
    )

    return X_scaled, y, scaler


def build_test_pipeline(scaler: StandardScaler) -> pd.DataFrame:
    """Preprocess test data using a previously fitted `scaler`.

    The test pipeline mirrors the train pipeline's feature engineering but
    does not touch targets. It is important to reuse the same `scaler` that
    was fit on training features to ensure consistent feature scaling.

    Returns the scaled test feature DataFrame indexed by `Exp`.
    """

    _, _, test_data = load_data()
    test_data = forward_fill_z_cols(test_data)
    X_test = aggregate_experiment_features(test_data)

    feature_cols = [c for c in X_test.columns if c != "Exp"]
    _, mapping = sanitize_feature_names(pd.Index(feature_cols))
    X_test = X_test.rename(columns=mapping)

    X_test = X_test.set_index("Exp")
    X_test_scaled = apply_scaler(X_test, scaler)

    return X_test_scaled


if __name__ == "__main__":
    X, y, scaler = build_train_pipeline()
    X_test = build_test_pipeline(scaler)

    print("Saved scaler to src/artifacts/")
