from pathlib import Path
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")


def load_data(data_dir: Path = DATA_DIR):

    train_data = pd.read_csv(data_dir / "datahow_interview_train_data.csv")
    train_targets = pd.read_csv(data_dir / "datahow_interview_train_targets.csv")
    test_data = pd.read_csv(data_dir / "datahow_interview_test_data.csv")

    return train_data, train_targets, test_data


def forward_fill_z_cols(df: pd.DataFrame):
    z_cols = [c for c in df.columns if c.startswith("Z:")]

    if not z_cols:
        return df
    df[z_cols] = df.groupby("Exp")[z_cols].ffill()

    return df


def aggregate_experiment_features(df: pd.DataFrame) -> pd.DataFrame:
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
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    arr = scaler.transform(X.values)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def build_train_pipeline() -> tuple[pd.DataFrame, pd.Series, StandardScaler]:

    train_data, train_targets, _ = load_data()
    train_data = forward_fill_z_cols(train_data)
    X = aggregate_experiment_features(train_data)

    y = train_targets.set_index("Exp")["Y:Titer"]
    X = X.set_index("Exp")
    X = X.join(y, how="left")
    y = X.pop("Y:Titer")

    scaler = fit_scaler(X)
    X_scaled = apply_scaler(X, scaler)

    with open("src/artifacts/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    return X_scaled, y, scaler


def build_test_pipeline(scaler: StandardScaler) -> pd.DataFrame:
    _, _, test_data = load_data()
    test_data = forward_fill_z_cols(test_data)
    X_test = aggregate_experiment_features(test_data).set_index("Exp")
    X_test_scaled = apply_scaler(X_test, scaler)

    return X_test_scaled


if __name__ == "__main__":
    X, y, scaler = build_train_pipeline()
    X_test = build_test_pipeline(scaler)

    print("Saved scaler to src/artifacts/")
