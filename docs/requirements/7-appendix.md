# 6. Appendix

---

## Supported Tickers (v1.0)
`^GSPC`, `AMZN`, `TSLA`, `NVDA`, `AAPL`, `GOOGL`, `MSFT`

---

## MLflow Model Registry Stage Transitions
```
None → Staging (after training) → Production (if best RMSE) → Archived (on next promotion)
```

---

## Drift Alert Threshold
Default: `drift_score > 0.5` (configurable via `.env` as `DRIFT_THRESHOLD`).
Evidently defines drift score as the share of features that have drifted.

---

## Glossary

| Term | Definition |
|---|---|
| **RMSE** | Root Mean Squared Error — primary model selection metric |
| **MAPE** | Mean Absolute Percentage Error — scale-independent accuracy measure |
| **Directional Accuracy** | % of predictions where predicted direction (up/down) matches actual |
| **Data Drift** | Shift in input feature distribution relative to the reference window |
| **Model Drift** | Degradation in prediction performance over time |
| **Reference Window** | Historical data snapshot used as the baseline for drift comparison |
