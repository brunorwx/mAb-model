# mAb Model

This repository contains the inference API for antibody titer prediction.
The Docker image is built with the preprocessing artifact baked in so the
container can run immediately after build without additional runtime
configuration. The final image does not include the raw `data/` folder.

## Build the Docker image

From the repository root:

```bash
docker build -t mab-model:latest .
```

## Run the default container

The default container command starts the FastAPI server using Uvicorn on port `8000`:

```bash
docker run --rm -p 8000:8000 mab-model:latest
```

Once running, the API is available at `http://localhost:8000`.

## Predict with the FastAPI server

Send a POST to `/predict` with JSON features:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "exp": 123,
    "features": {
      "Z_FeedStart": 1.0,
      "Z_FeedEnd": 0.5,
      "Z_FeedRateGlc": 0.2,
      "Z_FeedRateGln": 0.1,
      "Z_phStart": 7.2,
      "Z_phEnd": 7.0,
      "Z_phShift": 0.1,
      "Z_tempStart": 37.0,
      "Z_tempEnd": 36.5,
      "Z_tempShift": 0.5,
      "Z_Stir": 100.0,
      "Z_DO": 50.0,
      "Z_ExpDuration": 10.0,
      "W_temp_last": 36.8,
      "W_temp_mean": 36.7,
      "W_temp_std": 0.2,
      "W_pH_last": 7.1,
      "W_pH_mean": 7.05,
      "W_pH_std": 0.05,
      "W_FeedGlc_last": 10.0,
      "W_FeedGlc_mean": 9.5,
      "W_FeedGlc_std": 0.3,
      "W_FeedGln_last": 5.0,
      "W_FeedGln_mean": 4.8,
      "W_FeedGln_std": 0.2,
      "X_VCD_last": 1.2,
      "X_VCD_mean": 1.1,
      "X_VCD_std": 0.05,
      "X_Glc_last": 8.0,
      "X_Glc_mean": 7.8,
      "X_Glc_std": 0.4,
      "X_Gln_last": 3.5,
      "X_Gln_mean": 3.4,
      "X_Gln_std": 0.2,
      "X_Amm_last": 0.3,
      "X_Amm_mean": 0.28,
      "X_Amm_std": 0.02,
      "X_Lac_last": 6.0,
      "X_Lac_mean": 5.8,
      "X_Lac_std": 0.3,
      "X_Lysed_last": 0.01,
      "X_Lysed_mean": 0.009,
      "X_Lysed_std": 0.002
    }
  }'
```

## Notes

- The current Dockerfile installs the Python dependencies listed in `pyproject.toml`.
- The architecture document at `docs/ARCHITECTURE.md` describes the intended
  ML pipeline and inference service design.
- Feel free to use in-IDE test runner or `pytest` to run the unit tests in `tests/`. e.g.,

```bash
pytest tests/test_preprocessing.py
```
