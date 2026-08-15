# 2.1 Functional Requirements

---

## Data Ingestion

| ID | Requirement | Status |
|---|---|---|
| FR-DI-001 | Fetch OHLCV data for configurable tickers from Yahoo Finance via `yfinance` | Done |
| FR-DI-002 | Validate fetched data against a schema (column names, dtypes, no nulls in Close/Volume) | Done |
| FR-DI-003 | Save raw data to a data folder `data/raw/{ticker}_{date}.csv` | Done |
| FR-DI-004 | Save a reference window snapshot to `data/reference/` on first run | Done |
| FR-DI-005 | Ingestion flow shall be schedulable via Prefect and runnable manually | Done |

---

## Feature Engineering

| ID | Requirement | Status |
|---|---|---|
| FR-FE-001 | Compute lag features for Close price: 1, 5, 10, 20 days | Done |
| FR-FE-002 | Compute RSI (14-day), MACD (12/26/9), Bollinger Bands (20-day), and ATR (14-day) | Done |
| FR-FE-003 | Add calendar features: day-of-week, month, quarter | Done |
| FR-FE-004 | Feature toggles shall be configurable via `configs/features_config.yaml` | Done |
| FR-FE-005 | Save processed datasets to `data/processed/{ticker}_{date}.csv` | Done |

---

## Model Training

| ID | Requirement |
|---|---|
| FR-MT-001 | Train four model types: Linear Regression, XGBoost, LSTM (PyTorch), Prophet |
| FR-MT-002 | Log RMSE, MAE, MAPE, and directional accuracy to MLflow per run |
| FR-MT-003 | Log all hyperparameters from `configs/model_config.yaml` to MLflow |
| FR-MT-004 | Promote the model with the lowest validation RMSE to `Production` in MLflow Model Registry |
| FR-MT-005 | Transition previous Production model to `Archived` before promotion |

---

## Prediction API

| ID | Requirement |
|---|---|
| FR-API-001 | Expose `POST /predict` accepting `{"ticker": str, "horizon_days": int}` |
| FR-API-002 | Response shall include predicted prices, model name, model version, and timestamp |
| FR-API-003 | Expose `GET /health` returning service status and loaded model version |
| FR-API-004 | Expose `GET /drift-status` returning the latest drift score and alert flag |
| FR-API-005 | Load the MLflow `Production` model at startup |

---

## Drift Monitoring

| ID | Requirement |
|---|---|
| FR-DM-001 | Compare current data window against the reference snapshot using Evidently `DataDriftPreset` |
| FR-DM-002 | Evaluate prediction quality using Evidently `RegressionPerformancePreset` |
| FR-DM-003 | Save a timestamped HTML report to `monitoring/reports/` per run |
| FR-DM-004 | Log drift metrics (drift score, number of drifted features) to MLflow |
| FR-DM-005 | Raise a drift alert if drift score exceeds a configurable threshold |

---

## Streamlit Dashboard

| ID | Requirement |
|---|---|
| FR-UI-001 | Existing tabs (Stock Monitor, Correlation, Yearly Changes) remain functionally unchanged |
| FR-UI-002 | New **Predictions** tab displays multi-model forecasts via the FastAPI `/predict` endpoint |
| FR-UI-003 | New **Model Registry** tab displays MLflow experiment runs with key metrics |
| FR-UI-004 | New **Drift Reports** tab renders the latest Evidently HTML report |
