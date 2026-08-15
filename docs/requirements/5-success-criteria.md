# 4. Success Criteria

| ID | Criterion | Target |
|---|---|---|
| SC-001 | XGBoost RMSE on held-out test set | < 3% of mean Close price |
| SC-002 | Directional accuracy (up/down) of best model | ≥ 55% |
| SC-003 | `POST /predict` p95 latency | < 2 seconds |
| SC-004 | All services healthy after `docker-compose up` | < 3 minutes |
| SC-005 | Drift report generated for a 30-day window | < 60 seconds end-to-end |
| SC-006 | Test suite pass rate | 100% on `pytest tests/` |
| SC-007 | MLflow logs present for all 4 model types | After one training flow run |
