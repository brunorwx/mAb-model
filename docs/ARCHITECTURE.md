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
