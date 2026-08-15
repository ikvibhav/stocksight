# 1. Overview

**Project:** ML-Powered Stock Prediction & Monitoring System
**Version:** 1.1 | **Status:** Active Development

---

## 1.1 Purpose & Scope

Extend an existing Streamlit stock dashboard into a production-grade MLOps platform that predicts S&P500 stock prices and monitors model health over time.

**MLOps lifecycle covered:**

1. Data ingestion & validation
2. Feature engineering
3. Model training & experiment tracking
4. Model deployment & serving
5. Drift detection & monitoring
6. Automated retraining triggers

**Target audience:**

| Persona | Description |
|---|---|
| **End User** | Seeks stock price predictions and trend analysis via the Streamlit UI |
| **ML Engineer** | Trains, evaluates, and promotes models via MLflow and Prefect pipelines |
| **Operator** | Monitors pipeline health, drift alerts, and service availability |

**Baseline features (v0) — currently live at [stockmonitor-ikv.streamlit.app](https://stockmonitor-ikv.streamlit.app):**

| Feature | Description |
|---|---|
| Stock data fetch | OHLCV data from Yahoo Finance via `yfinance` for 7 configurable tickers |
| Moving averages | 50-day and 200-day MA plotted against closing price |
| Interactive charts | Toggle between static (matplotlib) and interactive (altair) charts |
| Configurable period | Time period selector: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max |
| Correlation heatmap | Heatmap of closing price correlations across selected tickers |
| Yearly % changes | Year-on-year percentage change table (5, 10, or 20-year lookback) |

**Limitation of v0 deployment:**
Streamlit Cloud spins down inactive apps requiring a manual wake-up, making it unsuitable for a persistent prediction service.

**Target state (v1) — two-phase deployment:**

**Phase a (Hybrid):**
1. ML backend — FastAPI, MLflow, Prefect, Postgres, MinIO deployed on a cloud VM.
2. Streamlit Cloud retained as the UI, calling the VM API over HTTPS.

**Phase b (Full VM):**
1. Streamlit service migrated to the same VM (Streamlit Cloud retired).
2. All services self-contained.

---

## 1.2 Goals & Non-Goals

**Goals:**
- Multi-model comparison with tracked metrics (RMSE, MAE, MAPE, directional accuracy)
- Automated model promotion: best model → MLflow `Production` stage
- REST endpoint for N-day ahead price predictions
- Scheduled drift monitoring with report generation
- All services containerized and runnable with `docker-compose up`

**Non-Goals (v1.0):**
- Real-time or intraday streaming data
- Authentication or authorisation on any service
- Financial advice or trading signal generation
- Assets outside equities (crypto, forex, commodities)
- Automated retraining triggered by drift alert *(planned for v2.0)*
- Streamlit Cloud as permanent deployment target *(retained as UI host in Phase 9a; retired in Phase 9b)*
- Managed cloud platform services (AWS RDS, GCP Vertex AI, Azure ML, etc.) — all services run self-hosted via Docker Compose on the VM

---

## 1.3 Architecture

```
Yahoo Finance API
       │
       ▼
[Prefect: Data Ingestion Flow]
       │
       ▼
[Feature Engineering] ──► data/processed/
       │
       ▼
[Prefect: Training Flow] ──► MLflow (Experiments + Model Registry)
                                        │
                              [Production Model]
                                        │
                                        ▼
                              [FastAPI Prediction Service]
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
               [Streamlit Dashboard]    [Prefect: Monitoring Flow]
                                                       │
                                               [Evidently Reports]
```

**Docker Compose services:**

| Service | Purpose | Port |
|---|---|---|
| `postgres` | Backend DB for MLflow + Prefect | 5432 |
| `minio` | Artifact store for MLflow | 9000 |
| `mlflow` | Experiment tracking + model registry | 5000 |
| `prefect` | Pipeline orchestration UI | 4200 |
| `api` | FastAPI prediction service | 8000 |
| `streamlit` | Existing UI (augmented) | 8501 |
