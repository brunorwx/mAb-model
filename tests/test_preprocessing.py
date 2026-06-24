import numpy as np
import pandas as pd
import numpy.testing as npt

from src.preprocessing import pipeline
from sklearn.preprocessing import StandardScaler


def test_forward_fill_z_cols():
    df = pd.DataFrame(
        {
            "Exp": [1, 1, 2, 2],
            "Time[day]": [0, 1, 0, 1],
            "Z:sp": [0.1, np.nan, 0.2, np.nan],
            "A": [10, 20, 30, 40],
        }
    )

    out = pipeline.forward_fill_z_cols(df.copy())

    assert out.loc[0, "Z:sp"] == 0.1
    assert out.loc[1, "Z:sp"] == 0.1
    assert out.loc[2, "Z:sp"] == 0.2
    assert out.loc[3, "Z:sp"] == 0.2


def test_aggregate_experiment_features():
    df = pd.DataFrame(
        {
            "Exp": [1, 1, 2],
            "RowID": [1, 2, 1],
            "Time[day]": [0, 1, 0],
            "Z:set": [0.5, 0.5, 0.7],
            "Val": [10.0, 14.0, 20.0],
        }
    )

    out = pipeline.aggregate_experiment_features(df)

    assert set(out["Exp"]) == {1, 2}

    row1 = out[out["Exp"] == 1].iloc[0]

    npt.assert_allclose(row1["Z:set"], 0.5)

    npt.assert_allclose(row1["Val_last"], 14.0)
    npt.assert_allclose(row1["Val_mean"], 12.0)
    npt.assert_allclose(row1["Val_std"], np.std([10.0, 14.0], ddof=1))


def test_fit_and_apply_scaler():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    scaler = pipeline.fit_scaler(X)
    Xs = pipeline.apply_scaler(X, scaler)

    npt.assert_allclose(Xs.mean().values, np.zeros_like(Xs.mean().values), atol=1e-7)
    npt.assert_allclose(
        Xs.std(ddof=0).values, np.ones_like(Xs.std(ddof=0).values), atol=1e-7
    )


def test_build_train_pipeline_monkeypatched(monkeypatch):

    train_data = pd.DataFrame(
        {
            "Exp": [1, 1, 2, 2],
            "Time[day]": [0, 1, 0, 1],
            "Z:sp": [0.1, np.nan, 0.2, np.nan],
            "Metric": [5.0, 7.0, 10.0, 12.0],
        }
    )
    train_targets = pd.DataFrame({"Exp": [1, 2], "Y:Titer": [0.9, 0.8]})
    test_data = pd.DataFrame(
        {"Exp": [3], "Time[day]": [0], "Z:sp": [0.3], "Metric": [6.0]}
    )

    def fake_load_data():
        return train_data.copy(), train_targets.copy(), test_data.copy()

    monkeypatch.setattr(pipeline, "load_data", fake_load_data)

    X_scaled, y, scaler = pipeline.build_train_pipeline()

    assert X_scaled.shape[0] == 2
    assert y.shape[0] == 2

    assert isinstance(scaler, StandardScaler)
