# 5. Delivery

---

## 5.1 Milestones

| Phase | Description | Key Deliverables |
|---|---|---|
| 1 | Infrastructure Setup | `docker-compose.yml`, `.env.example`, updated `requirements.txt` |
| 2 | Feature Engineering | `utils/feature_engineering.py`, `configs/features_config.yaml` |
| 3 | Data Pipeline | `pipelines/data_ingestion.py`, schema validation, `data/` directories |
| 4 | Model Training | 4 model files, `pipelines/training_pipeline.py`, `configs/model_config.yaml` |
| 5 | Inference API | `api/main.py`, `api/schemas.py`, all endpoints |
| 6 | Drift Monitoring | `monitoring/evidently_config.py`, `pipelines/monitoring_pipeline.py` |
| 7 | Dashboard Update | 3 new tabs in `app.py` (Predictions, Model Registry, Drift Reports) |
| 8 | Tests | `tests/test_data_pipeline.py`, `test_models.py`, `test_api.py` |
| 9a | Hybrid Deployment | Provision VM, deploy backend services (FastAPI, MLflow, Prefect, Postgres, MinIO) via Docker Compose, configure CORS on FastAPI, point Streamlit Cloud app at VM API |
| 9b | Full VM Migration | Add Streamlit service to Docker Compose, configure reverse proxy (Nginx/Caddy), set up DNS, retire Streamlit Cloud deployment |

Phases 1–3 are sequential. Phases 4 and 5 may run in parallel after Phase 3. Phase 6 may start in parallel with Phase 5. Phase 7 depends on Phases 5 and 6. Phase 8 runs alongside all phases.

---

## 5.2 Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-001 | `yfinance` API changes or rate-limits break ingestion | Medium | High | Pin version; add retry logic with exponential backoff; support CSV fallback |
| R-002 | Data leakage in feature engineering | Medium | High | Strict temporal train/val/test split; lag features reference only past timestamps |
| R-003 | LSTM overfitting on small datasets | High | Medium | Dropout, early stopping, cross-validation; compare against simpler baselines |
| R-004 | MLflow/Prefect backend not ready on first startup | Low | Medium | `depends_on` + health checks in Docker Compose; retry logic in pipeline startup |
| R-005 | MinIO misconfiguration causes model save failures | Low | High | Smoke test artifact upload; document bucket setup in `.env.example` |
| R-006 | Prophet installation conflicts with PyTorch dependencies | Medium | Low | Document known conflicts in `requirements.txt` comments |
| R-007 | Cloud VM free-tier resource limits constrain model training | Medium | Medium | Profile memory usage locally first; offload LSTM training to a separate step or reduce model size if needed |
| R-008 | CORS misconfiguration blocks Streamlit Cloud from calling VM API (Phase 9a) | Medium | High | Configure FastAPI `CORSMiddleware` with explicit Streamlit Cloud origin; test with browser dev tools before Phase 9b |
