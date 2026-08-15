.PHONY: help setup up down restart logs ps build clean-volumes \
        minio-init mlflow-ui prefect-ui streamlit-ui minio-ui

# ── Default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Stock Monitor — Available Commands"
	@echo "─────────────────────────────────────────────────"
	@echo "  make setup          Copy .env.example to .env"
	@echo "  make build          Build all custom Docker images"
	@echo "  make up             Start all Phase 1 services (detached)"
	@echo "  make down           Stop all services (keep data)"
	@echo "  make clean-volumes  Stop all services and delete all data"
	@echo "  make restart        Restart all services"
	@echo "  make ps             Show service status and health"
	@echo "  make logs           Tail logs for all services"
	@echo "  make logs s=mlflow  Tail logs for a specific service"
	@echo "  make minio-init     Create MLflow artifacts bucket in MinIO"
	@echo "  make mlflow-ui      Open MLflow UI in browser"
	@echo "  make prefect-ui     Open Prefect UI in browser"
	@echo "  make streamlit-ui   Open Streamlit UI in browser"
	@echo "  make minio-ui       Open MinIO UI in browser"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env created from .env.example — update passwords before deploying"; \
	else \
		echo ".env already exists, skipping"; \
	fi

precommit-setup:
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed"

python-venv:
	@if [ -d venv ]; then \
		echo "Virtual environment already exists, skipping"; \
	else \
		python3 -m venv venv --prompt=stockmonitorapp && \
		./venv/bin/pip install -r requirements.txt; \
	fi

python-venv-clean:
	rm -rf venv
	@echo "Virtual environment deleted"

python-update-requirements:
	./venv/bin/pip freeze > requirements.txt
	@echo "requirements.txt updated"

prefect:
	@echo "Installing Prefect CLI in Python virtual environment..."
	./venv/bin/pip install prefect
	@echo "Prefect CLI installed"
	./venv/bin/prefect config set PREFECT_API_URL=http://localhost:4200/api
	@echo "Prefect API URL set to http://localhost:4200/api"
	./venv/bin/prefect work-pool inspect local-process >/dev/null 2>&1 || \
	./venv/bin/prefect work-pool create local-process --type process
	@echo "Prefect work pool 'local-process' ready"
	./venv/bin/prefect deploy --all
	@echo "Prefect flows deployed"

prefect-worker:
	./venv/bin/prefect worker start --pool local-process

# ── Test ─────────────────────────────────────────────────────────────────────
pytest:
	./venv/bin/pytest tests/
# ── Lifecycle ─────────────────────────────────────────────────────────────────
build:
	docker compose build
up:
	docker compose up -d postgres minio mlflow prefect streamlit

down:
	docker compose down

clean-volumes:
	docker compose down -v
	@echo "All services stopped and data volumes deleted"

restart:
	docker compose restart

check-images:
	docker compose images

# ── Observability ─────────────────────────────────────────────────────────────
ps:
	docker compose ps

# Usage: make logs         (all services)
#        make logs s=mlflow (single service)
logs:
	docker compose logs -f $(s)

# ── MinIO Init ────────────────────────────────────────────────────────────────
minio-init:
	@echo "Creating MLflow artifacts bucket in MinIO..."
	docker compose exec minio mc alias set local http://localhost:9000 \
		$$(grep MINIO_ROOT_USER .env | cut -d= -f2) \
		$$(grep MINIO_ROOT_PASSWORD .env | cut -d= -f2)
	docker compose exec minio mc mb --ignore-existing local/$$(grep MLFLOW_BUCKET .env | cut -d= -f2)
	@echo "Bucket ready"

# ── Open UIs ──────────────────────────────────────────────────────────────────
mlflow-ui:
	xdg-open http://localhost:5000

prefect-ui:
	xdg-open http://localhost:4200

streamlit-ui:
	xdg-open http://localhost:8501

minio-ui:
	xdg-open http://localhost:9000
