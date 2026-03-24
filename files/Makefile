# ─────────────────────────────────────────────────────────────────────────────
# Makefile — Research Agent
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install run test lint format security docker-build docker-up docker-down

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:       ## Create venv and install all dependencies
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install ruff black bandit safety
	@echo "✅ Done. Activate with: source venv/bin/activate"

run:           ## Start the Streamlit app
	streamlit run src/app.py

test:          ## Run unit tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing -m "not integration"

test-all:      ## Run ALL tests including integration (needs internet)
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:          ## Run ruff linter
	ruff check src/ tests/

format:        ## Auto-format code with black
	black src/ tests/

format-check:  ## Check formatting without changing files (used in CI)
	black --check src/ tests/

security:      ## Run bandit (SAST) + safety (CVE check)
	@echo "── Bandit (static security scan) ──"
	bandit -r src/ -ll -ii
	@echo "\n── Safety (dependency CVEs) ──"
	safety check --file requirements.txt

docker-build:  ## Build Docker image
	docker build -t research-agent:latest .

docker-up:     ## Start app + Ollama via Docker Compose
	docker compose up --build

docker-down:   ## Stop all containers
	docker compose down

pull-model:    ## Pull default LLM model into Ollama (run after docker-up)
	docker compose exec ollama ollama pull llama3

clean:         ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/
