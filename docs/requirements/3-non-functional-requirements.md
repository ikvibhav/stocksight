# 2.2 Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-001 | Performance | `POST /predict` p95 response time < 2 seconds for horizon ≤ 30 days |
| NFR-002 | Performance | Streamlit dashboard initial load < 5 seconds |
| NFR-003 | Reliability | All Docker Compose services shall define health checks |
| NFR-004 | Reproducibility | Training runs shall be fully reproducible given the same data and config (fixed random seeds) |
| NFR-005 | Data Freshness | Ingestion pipeline supports daily scheduling; data older than 1 day triggers a staleness warning |
| NFR-006 | Portability | The full system runs via `docker-compose up` on both local machines and a cloud VM — no prerequisites beyond Docker |
| NFR-007 | Testability | All pipeline stages and API endpoints shall have automated tests in `tests/` |

---

# 2.3 Technical Stack

**Runtime:** Python 3.10, Docker 24.x+, Docker Compose v2+

| Component | Technology | Version |
|---|---|---|
| Dashboard | Streamlit | 1.38.x |
| REST API | FastAPI + Uvicorn | 0.115.x |
| Data Source | yfinance | 0.2.x |
| ML — Baseline | scikit-learn (Linear Regression) | 1.5.x |
| ML — Gradient Boosting | XGBoost | 2.x |
| ML — Deep Learning | PyTorch (LSTM) | 2.x |
| ML — Time Series | Prophet | 1.1.x |
| Experiment Tracking | MLflow | 2.x |
| Orchestration | Prefect | 3.x |
| Drift Monitoring | Evidently AI | 0.4.x |
| Schema Validation | Pandera | 0.20.x |
| Backend DB | PostgreSQL | 15 |
| Artifact Store | MinIO | latest |

**Configuration:**
- Hyperparameters: `configs/model_config.yaml`
- Feature toggles: `configs/features_config.yaml`
- Service connection strings: `.env` (see `.env.example`)
