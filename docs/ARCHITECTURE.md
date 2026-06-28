# First idea of the architecture

## Initial high-Level overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          mAb Titer Prediction System                        │
│                                                                             │
│  ┌──────────────────────────────┐    ┌───────────────────────────────────┐  │
│  │      Part 1: ML Pipeline     │    │    Part 2: Inference Service      │  │
│  │                              │    │                                   │  │
│  │  Data → Preprocess → Train   │───▶│  FastAPI Server + Saved Model     │  │
│  │            → Evaluate        │    │                                   │  │
│  └──────────────────────────────┘    └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project structure

The repository is organized to separate data processing, model training, inference, and API wiring.

- `Dockerfile`: multi-stage build that installs dependencies, trains models in a builder stage, and copies only code plus generated artifacts into the runtime image.
- `docker-compose.yml`: optional orchestration configuration for containerized deployment.
- `data/`: raw input CSVs used by the training pipeline. These files are kept outside of runtime unless explicitly needed in the build stage.
- `docs/`: design documentation, architecture rationale, and data descriptions.
- `src/api`: FastAPI application code, request/response DTOs, routes, and thin service-layer adapters.
- `src/model`: training, model evaluation, and inference artifact loading.
- `src/preprocessing`: data loading, feature engineering, scaling logic, and preprocessor serialization.
- `src/artifacts`: generated model and preprocessor artifacts such as `xgb_model.pkl`, `lgbm_model.pkl`, and `scaler_preprocessor.pkl`.
- `tests/`: unit tests verifying preprocessing behavior and model/inference logic.

## Why this layout

This structure is intentionally designed around separation of concerns and production readiness.

- `src/api` is isolated from model and preprocessing internals so API behavior, request validation, and routing remain independent of training details.
- `src/model` contains training and inference logic, keeping model-specific code separate from data transformation logic.
- `src/preprocessing` centralizes the feature pipeline so the same transformations can be reused in both training and inference.
- `src/api/services/prediction.py` acts as a thin adapter between the API and the model layer, making artifact path injection and testing easier.
- `src/artifacts` stores serialized objects rather than embedding them in source modules. This keeps code clean while allowing the Docker build to copy only the necessary artifacts into runtime.
- Raw data lives in `data/` to clearly distinguish training inputs from deployable application code. The runtime image does not need the raw CSVs once artifacts are produced.
- The `Dockerfile` trains models during build, then copies the trained artifacts into a smaller runtime image. This makes the deployed container self-sufficient and avoids training at inference time.
- Pydantic models in `src/api/dto` ensure valid request payloads and consistent response shapes.
- The FastAPI lifespan hook loads model artifacts once at startup, avoiding repeated load overhead on every request.

## Part 1: ML Pipeline Flow

```
 ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
 │  CSV Files   │     │  Preprocessing   │     │   Model Training │
 │              │     │                  │     │                  │
 │ train_data   │────▶│ 1. Load & parse  │────▶│ 1. Train/val     │
 │ train_targets│     │ 2. Handle Z NaNs │     │    split         │
 │ test_data    │     │ 3. Feature eng.  │     │ 2. Fit model(s)  │
 │              │     │ 4. Aggregate     │     │ 3. Hyperparam    │
 └──────────────┘     │    time series   │     │    tuning        │
                      │ 5. Scale/norm    │     └────────┬─────────┘
                      └──────────────────┘              │
                                                        ▼
                      ┌──────────────────┐     ┌──────────────────┐
                      │    Artifacts     │     │   Evaluation     │
                      │                  │     │                  │
                      │ models/          │◀────│ 1. Metrics       │
                      │  model.joblib    │     │    (R², RMSE,    │
                      │  scaler.joblib   │     │     MAE)         │
                      │  pipeline.joblib │     │ 2. Cross-val     │
                      │                  │     │ 3. Residual      │
                      └──────────────────┘     │    analysis      │
                                               │ 4. Test predict  │
                                               └──────────────────┘
```

## Part 2: Inference Service

```
                   ┌─────────────────────────────────────────────┐
                   │             FastAPI Application             │
                   │                                             │
  ┌──────────┐     │  ┌─────────┐    ┌────────────────────────┐  │
  │  Client  │────▶│  │ Routes  │    │     Model Service      │  │
  │ (HTTP)   │     │  │         │    │                        │  │
  │          │     │  │ GET     │    │ ┌──────────────────┐   │  │
  │          │     │  │ /health │───▶│ │ Health check     │   │  │
  │          │     │  │         │    │ └──────────────────┘   │  │
  │          │     │  │         │    │                        │  │
  │          │     │  │ POST    │    │ ┌──────────────────┐   │  │
  │          │     │  │ /predict│───▶│ │ 1. Validate DTO  │   │  │
  │          │◀────│  │         │    │ │ 2. Preprocess    │   │  │
  │          │     │  │         │    │ │ 3. Run model     │   │  │
  │          │     │  └─────────┘    │ │ 4. Return titer  │   │  │
  │          │     │                 │ └──────────────────┘   │  │
  └──────────┘     │                 └───────────┬────────────┘  │
                   │                             │               │
                   │                  ┌──────────┴──────────┐    │
                   │                  │  Saved Artifacts    │    │
                   │                  │  (loaded at startup)│    │
                   │                  │                     │    │
                   │                  │  - model.joblib     │    │
                   │                  │  - scaler.joblib    │    │
                   │                  │  - pipeline.joblib  │    │
                   │                  └─────────────────────┘    │
                   └─────────────────────────────────────────────┘
```

