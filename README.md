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
  -H 'Content-Type: application/json' \
  -d '{"exp": 123, "features": {"a": 1.0, "b": 2.0}}'
```

## Notes

- The current Dockerfile installs the Python dependencies listed in `pyproject.toml`.
- The architecture document at `docs/ARCHITECTURE.md` describes the intended
  ML pipeline and inference service design.
- Feel free to use in-IDE test runner or `pytest` to run the unit tests in `tests/`. e.g.,

```bash
pytest tests/test_preprocessing.py
```
