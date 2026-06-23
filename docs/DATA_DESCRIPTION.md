# Data Description

## Overview

This dataset contains simulated upstream bioprocess data for **monoclonal antibody (mAb) production**. Each experiment represents a bioreactor run where cells (typically CHO cells) are cultured under controlled conditions to produce mAb. The goal is to predict the **final product titer** (mAb concentration) at the end of each experiment.

## Files

| File                                          | Description                                        | Shape                 |
| --------------------------------------------- | -------------------------------------------------- | --------------------- |
| `datahow_interview_train_data.csv`            | Training data inputs (time-series + scalars)       | 990 rows x 26 columns |
| `datahow_interview_train_targets.csv`         | Training data targets (final titer per experiment) | 100 rows x 4 columns  |
| `datahow_interview_test_data.csv`             | Test data inputs                                   | 300 rows x 26 columns |
| `datahow_interview_test_targets-TEMPLATE.csv` | Placeholder for test targets (all set to 2000)     | 20 rows x 4 columns   |
| `inference_server_spec.yml`                   | OpenAPI spec for the inference microservice        | -                     |

## Dataset Structure

- **Training set**: 100 experiments with variable durations (7-14 days), totaling 990 time-step rows.
- **Test set**: 20 experiments, all with 14-day duration (complete time series), totaling 300 rows.
- Each row represents a single time point (day) within an experiment.
- The target is a **single scalar value per experiment**: the final mAb titer.

### Experiment Duration Distribution (Training)

| Duration | Number of Experiments |
| -------- | --------------------- |
| 7 days   | 30                    |
| 8 days   | 20                    |
| 9 days   | 20                    |
| 10 days  | 20                    |
| 14 days  | 10                    |

## Column Descriptions

### Identifiers

| Column      | Description                                                        |
| ----------- | ------------------------------------------------------------------ |
| `RowID`     | Global row index across all experiments                            |
| `Exp`       | Experiment identifier (e.g., "Exp 1", "Test Exp 1")                |
| `Time[day]` | Time point in days from the start of the experiment (0, 1, 2, ...) |

### Z-prefix: Scalar Process Parameters (Set Points)

These are **fixed design parameters** set before or during the experiment. They appear only on the first row (day 0) of each experiment and are `NaN` for subsequent time points.

| Column          | Description                                         | Range           |
| --------------- | --------------------------------------------------- | --------------- |
| `Z:FeedStart`   | Day when nutrient feeding begins                    | 1 - 3           |
| `Z:FeedEnd`     | Day when nutrient feeding ends                      | 9 - 13          |
| `Z:FeedRateGlc` | Glucose feed rate                                   | 2.02 - 5.98     |
| `Z:FeedRateGln` | Glutamine feed rate                                 | 6.01 - 7.99     |
| `Z:phStart`     | Initial pH set point                                | 6.51 - 7.49     |
| `Z:phEnd`       | Final pH set point (after pH shift)                 | 6.01 - 6.99     |
| `Z:phShift`     | Day when pH shifts from start to end value          | 6 - 14          |
| `Z:tempStart`   | Initial temperature set point (C)                   | 36.01 - 37.99   |
| `Z:tempEnd`     | Final temperature set point (C)                     | 35.01 - 36.99   |
| `Z:tempShift`   | Day when temperature shifts from start to end value | 6 - 14          |
| `Z:Stir`        | Stirring speed (RPM)                                | 150.51 - 249.49 |
| `Z:DO`          | Dissolved oxygen set point (%)                      | 30.25 - 79.75   |
| `Z:ExpDuration` | Total experiment duration in days                   | 7 - 14          |

### W-prefix: Time-Series Input Profiles (Controlled Variables)

These are **actuated process inputs** that vary over time based on the scalar set points (Z parameters). They have a value at every time point.

| Column      | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| `W:temp`    | Bioreactor temperature (C) at each time point                          |
| `W:pH`      | Bioreactor pH at each time point                                       |
| `W:FeedGlc` | Glucose feed applied at each time point (0 when outside feed window)   |
| `W:FeedGln` | Glutamine feed applied at each time point (0 when outside feed window) |

### X-prefix: Time-Series Observations (Measured Variables)

These are **measured state variables** reflecting the biological response of the culture. They have a value at every time point.

| Column    | Description                                                |
| --------- | ---------------------------------------------------------- |
| `X:VCD`   | Viable Cell Density (10^6 cells/mL)                        |
| `X:Glc`   | Glucose concentration (g/L)                                |
| `X:Gln`   | Glutamine concentration (mmol/L)                           |
| `X:Amm`   | Ammonia concentration (mmol/L) - toxic metabolic byproduct |
| `X:Lac`   | Lactate concentration (g/L) - metabolic byproduct          |
| `X:Lysed` | Lysed (dead) cell fraction                                 |

### Y-prefix: Target Variable

| Column    | Description                            | Range            |
| --------- | -------------------------------------- | ---------------- |
| `Y:Titer` | Final monoclonal antibody titer (mg/L) | 283.46 - 4822.70 |

## Target Statistics (Training Set)

| Statistic | Value        |
| --------- | ------------ |
| Mean      | 1314.51 mg/L |
| Std Dev   | 753.09 mg/L  |
| Min       | 283.46 mg/L  |
| Max       | 4822.70 mg/L |

## Key Data Characteristics

1. **Variable-length time series**: Training experiments range from 7 to 14 days; test experiments are all 14 days. Models must handle sequences of different lengths.

2. **Scalar parameters only at day 0**: The Z-columns are recorded only on the first row of each experiment. They need to be forward-filled or extracted separately during preprocessing.

3. **Feeding window**: Glucose and glutamine feeds (`W:FeedGlc`, `W:FeedGln`) are active only between `Z:FeedStart` and `Z:FeedEnd` days, set to 0 outside this window.

4. **Temperature and pH shifts**: Temperature and pH shift from their start values to their end values at the specified shift day, creating a step-change profile in the `W:temp` and `W:pH` columns.

5. **Biological dynamics**: Cells grow (increasing `X:VCD`), consume nutrients (`X:Glc`, `X:Gln` deplete), and produce metabolic byproducts (`X:Amm`, `X:Lac` accumulate). Cell death (`X:Lysed`) increases toward the end of culture.

6. **One target per experiment**: Despite multiple time-step rows per experiment, the target `Y:Titer` is a single final value per experiment.
