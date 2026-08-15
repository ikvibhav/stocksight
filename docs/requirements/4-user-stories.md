# 3. User Stories

---

## End User
- **US-01:** As an end user, I want to select a ticker and see predicted closing prices for the next 5–30 days.
- **US-02:** As an end user, I want to see which model produced the prediction and its reported accuracy.
- **US-03:** As an end user, I want existing charts (moving averages, correlation, yearly changes) unchanged.

## ML Engineer
- **US-04:** As an ML engineer, I want to trigger a training run for all models from the Prefect UI.
- **US-05:** As an ML engineer, I want to compare model metrics across runs in MLflow.
- **US-06:** As an ML engineer, I want the best-performing model automatically promoted to `Production`.
- **US-07:** As an ML engineer, I want to configure model hyperparameters via `configs/model_config.yaml` without touching code.

## Operator
- **US-08:** As an operator, I want a drift report showing whether input feature distributions have shifted.
- **US-09:** As an operator, I want a `/health` endpoint to check API service status.
- **US-10:** As an operator, I want all services to start with a single `docker-compose up` command.
