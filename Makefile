.PHONY: help install setup test run docker-up docker-down

help:
	@echo "Swing Trading System - Makefile"
	@echo ""
	@echo "Commands:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup        - Set up database and initial config"
	@echo "  make test         - Run tests"
	@echo "  make run          - Start Lightning App (API + Training)"
	@echo "  make docker-up    - Start TimescaleDB with Docker"
	@echo "  make docker-down  - Stop TimescaleDB"

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

run:
	lightning run app app.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

