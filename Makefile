.PHONY: help install setup test run-api run-ui docker-up docker-down

help:
	@echo "Swing Trading System - Makefile"
	@echo ""
	@echo "Commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup        - Set up database and initial config"
	@echo "  make test         - Run tests"
	@echo "  make run-api      - Start FastAPI server"
	@echo "  make run-ui       - Start React development server"
	@echo "  make docker-up    - Start all services with Docker Compose"
	@echo "  make docker-down  - Stop all services"

install:
	pip install -r requirements.txt

setup:
	@echo "Setting up database..."
	@if [ ! -f config/config.yaml ]; then \
		cp config/config.example.yaml config/config.yaml; \
		echo "Created config/config.yaml - please update with your API keys"; \
	fi
	@echo "Run: psql -U postgres -d swing_trading < scripts/schema.sql"

test:
	pytest tests/

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	cd src/ui && npm install && npm start

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

