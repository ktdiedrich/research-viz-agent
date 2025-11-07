.PHONY: help install install-dev test test-coverage test-fast clean clean-all lint format check serve docs build
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Variables
PYTHON := python
UV := uv
PYTEST := pytest
VENV := .venv
SRC := research_viz_agent
TESTS := tests
DOCS := docs

help: ## Show this help message
	@echo "$(BLUE)research-viz-agent Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# Installation targets
install: ## Install the package in production mode
	@echo "$(BLUE)Installing research-viz-agent with uv...$(NC)"
	$(UV) pip install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install the package with development dependencies
	@echo "$(BLUE)Installing research-viz-agent with dev dependencies (uv)...$(NC)"
	$(UV) pip install -e ".[dev]"
	@echo "$(GREEN)✓ Development installation complete$(NC)"

# Testing targets
test: ## Run all tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TESTS) -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-fast: ## Run tests without coverage (faster)
	@echo "$(BLUE)Running tests (fast mode)...$(NC)"
	$(PYTEST) $(TESTS) -v --no-cov
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-coverage: ## Run tests and open HTML coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TESTS) -v
	@echo "$(BLUE)Opening coverage report...$(NC)"
	@if command -v xdg-open > /dev/null; then \
		xdg-open htmlcov/index.html; \
	elif command -v open > /dev/null; then \
		open htmlcov/index.html; \
	else \
		echo "$(YELLOW)Please open htmlcov/index.html manually$(NC)"; \
	fi
	@echo "$(GREEN)✓ Coverage report generated$(NC)"

test-unit: ## Run only unit tests (excluding integration)
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) $(TESTS) -v -m "not integration" --no-cov
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration: ## Run only integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) $(TESTS) -v -m "integration" --no-cov
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-agent-protocol: ## Run agent protocol tests
	@echo "$(BLUE)Running agent protocol tests...$(NC)"
	$(PYTEST) $(TESTS)/test_agent_protocol.py $(TESTS)/test_agent_server.py -v
	@echo "$(GREEN)✓ Agent protocol tests complete$(NC)"

test-mcp: ## Run MCP tools tests
	@echo "$(BLUE)Running MCP tools tests...$(NC)"
	$(PYTEST) $(TESTS)/test_arxiv_tool.py $(TESTS)/test_pubmed_tool.py $(TESTS)/test_huggingface_tool.py -v --no-cov
	@echo "$(GREEN)✓ MCP tools tests complete$(NC)"

test-cli: ## Run CLI tests
	@echo "$(BLUE)Running CLI tests...$(NC)"
	$(PYTEST) $(TESTS)/test_cli.py $(TESTS)/test_cli_extended.py -v
	@echo "$(GREEN)✓ CLI tests complete$(NC)"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	@if command -v ptw > /dev/null; then \
		ptw -- $(TESTS) -v --no-cov; \
	else \
		echo "$(RED)Error: pytest-watch not installed. Install with: pip install pytest-watch$(NC)"; \
		exit 1; \
	fi

# Code quality targets
lint: ## Run linting checks (ruff)
	@echo "$(BLUE)Running linter...$(NC)"
	@if command -v ruff > /dev/null; then \
		ruff check $(SRC) $(TESTS); \
		echo "$(GREEN)✓ Linting complete$(NC)"; \
	else \
		echo "$(YELLOW)Warning: ruff not installed. Install with: pip install ruff$(NC)"; \
	fi

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	@if command -v ruff > /dev/null; then \
		ruff format $(SRC) $(TESTS); \
		echo "$(GREEN)✓ Formatting complete$(NC)"; \
	else \
		echo "$(YELLOW)Warning: ruff not installed. Install with: pip install ruff$(NC)"; \
	fi

format-check: ## Check code formatting without modifying files
	@echo "$(BLUE)Checking code formatting...$(NC)"
	@if command -v ruff > /dev/null; then \
		ruff format --check $(SRC) $(TESTS); \
		echo "$(GREEN)✓ Format check complete$(NC)"; \
	else \
		echo "$(YELLOW)Warning: ruff not installed. Install with: pip install ruff$(NC)"; \
	fi

check: format-check lint test-fast ## Run all checks (format, lint, tests)
	@echo "$(GREEN)✓ All checks passed$(NC)"

# Cleaning targets
clean: ## Remove build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-coverage: ## Remove coverage reports
	@echo "$(BLUE)Cleaning coverage reports...$(NC)"
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	@echo "$(GREEN)✓ Coverage cleanup complete$(NC)"

clean-rag: ## Remove RAG database directories
	@echo "$(BLUE)Cleaning RAG databases...$(NC)"
	@echo "$(YELLOW)Warning: This will delete all RAG databases!$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf chroma_db* 2>/dev/null || true; \
		echo "$(GREEN)✓ RAG databases removed$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-all: clean clean-coverage ## Remove all generated files (build, cache, coverage)
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

# Server targets
up: ## Start the agent server (default: port 8000)
	@echo "$(BLUE)Starting agent server...$(NC)"
	research-viz-agent serve

up-dev: ## Start the agent server with auto-reload
	@echo "$(BLUE)Starting agent server in development mode...$(NC)"
	uvicorn research_viz_agent.agent_protocol.server:app --reload --host 0.0.0.0 --port 8000

up-custom: ## Start server with custom port (use: make up-custom PORT=9000)
	@echo "$(BLUE)Starting agent server on port $(PORT)...$(NC)"
	research-viz-agent serve --port $(PORT)

up-background: ## Start server in background (saves PID to .server.pid)
	@echo "$(BLUE)Starting agent server in background...$(NC)"
	@nohup research-viz-agent serve > server.log 2>&1 & echo $$! > .server.pid
	@echo "$(GREEN)✓ Server started (PID: $$(cat .server.pid))$(NC)"
	@echo "$(YELLOW)Logs: tail -f server.log$(NC)"

down: ## Stop background server
	@if [ -f .server.pid ]; then \
		PID=$$(cat .server.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "$(BLUE)Stopping server (PID: $$PID)...$(NC)"; \
			kill $$PID; \
			rm .server.pid; \
			echo "$(GREEN)✓ Server stopped$(NC)"; \
		else \
			echo "$(YELLOW)Server not running (stale PID file removed)$(NC)"; \
			rm .server.pid; \
		fi; \
	else \
		echo "$(YELLOW)No server PID file found. Attempting to kill by port...$(NC)"; \
		if command -v lsof > /dev/null; then \
			PID=$$(lsof -ti:8000); \
			if [ -n "$$PID" ]; then \
				echo "$(BLUE)Found process on port 8000 (PID: $$PID)$(NC)"; \
				kill $$PID; \
				echo "$(GREEN)✓ Server stopped$(NC)"; \
			else \
				echo "$(YELLOW)No server running on port 8000$(NC)"; \
			fi; \
		else \
			echo "$(RED)lsof not available. Cannot detect server.$(NC)"; \
			echo "$(YELLOW)Try: pkill -f 'research-viz-agent serve'$(NC)"; \
		fi; \
	fi

restart: down up-background ## Restart the server (stop + start in background)

# Research targets
research: ## Run a quick research query (use: make research QUERY="lung cancer detection")
	@if [ -z "$(QUERY)" ]; then \
		echo "$(RED)Error: QUERY not specified. Usage: make research QUERY=\"your query\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running research query: $(QUERY)$(NC)"
	research-viz-agent "$(QUERY)"

research-no-llm: ## Run research without LLM summary (use: make research-no-llm QUERY="...")
	@if [ -z "$(QUERY)" ]; then \
		echo "$(RED)Error: QUERY not specified. Usage: make research-no-llm QUERY=\"your query\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running research query (no LLM): $(QUERY)$(NC)"
	research-viz-agent "$(QUERY)" --llm-provider none

rag-stats: ## Show RAG database statistics
	@echo "$(BLUE)RAG Database Statistics:$(NC)"
	research-viz-agent --rag-stats

rag-tracking: ## Show RAG tracking summary
	@echo "$(BLUE)RAG Tracking Summary:$(NC)"
	research-viz-agent --show-tracking

# Documentation targets
docs-serve: ## Serve documentation locally (requires mkdocs)
	@echo "$(BLUE)Serving documentation...$(NC)"
	@if command -v mkdocs > /dev/null; then \
		mkdocs serve; \
	else \
		echo "$(YELLOW)mkdocs not installed. Opening docs directory...$(NC)"; \
		@if command -v xdg-open > /dev/null; then \
			xdg-open $(DOCS)/; \
		elif command -v open > /dev/null; then \
			open $(DOCS)/; \
		fi; \
	fi

docs-list: ## List all documentation files
	@echo "$(BLUE)Documentation files:$(NC)"
	@ls -1 $(DOCS)/*.md | while read file; do \
		basename=`basename $$file`; \
		printf "  $(YELLOW)%-40s$(NC)" "$$basename"; \
		head -n 1 $$file | sed 's/^# //' | sed 's/^## //'; \
	done

# Build targets
build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Build complete$(NC)"

build-check: ## Check if package can be built
	@echo "$(BLUE)Checking package build...$(NC)"
	$(PYTHON) -m build --sdist --wheel --outdir dist/test/
	rm -rf dist/test/
	@echo "$(GREEN)✓ Package build check passed$(NC)"

# Development workflow targets
dev-setup: install-dev ## Complete development setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env file from example...$(NC)"; \
		cp .env.example .env; \
		echo "$(YELLOW)⚠ Please edit .env and add your API keys$(NC)"; \
	fi
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Edit .env and add your API keys (GITHUB_TOKEN or OPENAI_API_KEY)"
	@echo "  2. Run 'make test' to verify installation"
	@echo "  3. Run 'make research QUERY=\"test\"' to try a search"

dev-reset: clean-all ## Reset development environment (clean all + reinstall)
	@echo "$(BLUE)Resetting development environment...$(NC)"
	$(MAKE) install-dev
	@echo "$(GREEN)✓ Development environment reset$(NC)"

# Examples targets
run-examples: ## Run all standalone example scripts (excludes client/orchestration)
	@echo "$(BLUE)Running standalone example scripts...$(NC)"
	@echo "$(YELLOW)Note: Skipping client/orchestration examples (require running server)$(NC)"
	@for script in examples/*.py; do \
		case "$$script" in \
			*agent_client.py|*agent_orchestration.py|*agent_server.py) \
				echo "$(YELLOW)Skipping $$script (requires server)$(NC)"; \
				;; \
			*) \
				echo "$(BLUE)Running $$script...$(NC)"; \
				$(PYTHON) $$script || true; \
				echo ""; \
				;; \
		esac; \
	done
	@echo "$(GREEN)✓ Examples complete$(NC)"
	@echo ""
	@echo "$(BLUE)To run client examples:$(NC)"
	@echo "  1. Start server: make example-server (or make up-background)"
	@echo "  2. Run client: make example-client"
	@echo "  3. Run orchestration: make example-orchestration"

example-server: ## Run the agent server example
	@echo "$(BLUE)Running agent server example...$(NC)"
	$(PYTHON) examples/agent_server.py

example-client: ## Run the agent client example (requires running server)
	@echo "$(BLUE)Running agent client example...$(NC)"
	@echo "$(YELLOW)Note: Make sure server is running (make up-background or make example-server)$(NC)"
	$(PYTHON) examples/agent_client.py

example-orchestration: ## Run the agent orchestration example (requires running server)
	@echo "$(BLUE)Running agent orchestration example...$(NC)"
	@echo "$(YELLOW)Note: Make sure server is running (make up-background or make example-server)$(NC)"
	$(PYTHON) examples/agent_orchestration.py

# Utility targets
version: ## Show version information
	@echo "$(BLUE)research-viz-agent version information:$(NC)"
	@grep "version" pyproject.toml | head -1 | awk -F'"' '{print "  Version: " $$2}'
	@echo "  Python: $$($(PYTHON) --version | awk '{print $$2}')"
	@echo "  Location: $$(pwd)"

env-check: ## Check environment and dependencies
	@echo "$(BLUE)Environment Check:$(NC)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@echo "uv: $$($(UV) --version 2>&1 | head -1)"
	@if [ -f .env ]; then \
		echo "$(GREEN)✓ .env file exists$(NC)"; \
	else \
		echo "$(RED)✗ .env file missing$(NC)"; \
	fi
	@if [ -d $(VENV) ]; then \
		echo "$(GREEN)✓ Virtual environment exists$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Virtual environment not found$(NC)"; \
	fi
	@echo ""
	@echo "$(BLUE)API Keys:$(NC)"
	@if [ -f .env ]; then \
		if grep -q "GITHUB_TOKEN=" .env && ! grep -q "GITHUB_TOKEN=\"\"" .env && ! grep -q "GITHUB_TOKEN=''" .env; then \
			echo "$(GREEN)✓ GITHUB_TOKEN configured$(NC)"; \
		else \
			echo "$(YELLOW)⚠ GITHUB_TOKEN not configured$(NC)"; \
		fi; \
		if grep -q "OPENAI_API_KEY=" .env && ! grep -q "OPENAI_API_KEY=\"\"" .env && ! grep -q "OPENAI_API_KEY=''" .env; then \
			echo "$(GREEN)✓ OPENAI_API_KEY configured$(NC)"; \
		else \
			echo "$(YELLOW)⚠ OPENAI_API_KEY not configured$(NC)"; \
		fi; \
	fi
	@echo ""
	@echo "$(BLUE)Running Services:$(NC)"
	@if [ -f .server.pid ]; then \
		PID=$$(cat .server.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "$(GREEN)✓ Agent server running (PID: $$PID, port 8000)$(NC)"; \
		else \
			echo "$(RED)✗ Server PID file exists but process not running$(NC)"; \
		fi; \
	else \
		if command -v lsof > /dev/null 2>&1; then \
			PORT_8000=$$(lsof -ti:8000 2>/dev/null); \
			if [ -n "$$PORT_8000" ]; then \
				echo "$(YELLOW)⚠ Process running on port 8000 (PID: $$PORT_8000) but no .server.pid file$(NC)"; \
			else \
				echo "$(YELLOW)⚠ No agent server running$(NC)"; \
			fi; \
		else \
			echo "$(YELLOW)⚠ Cannot check for running server (lsof not available)$(NC)"; \
		fi; \
	fi

status: env-check ## Show project status (alias for env-check)

list-models: ## List available LLM models
	@echo "$(BLUE)Available LLM Models:$(NC)"
	@echo ""
	@echo "$(YELLOW)GitHub Models:$(NC)"
	research-viz-agent --list-models github || true
	@echo ""
	@echo "$(YELLOW)OpenAI Models:$(NC)"
	research-viz-agent --list-models openai || true

info: ## Show comprehensive project information
	@$(MAKE) version
	@echo ""
	@$(MAKE) env-check
	@echo ""
	@echo "$(BLUE)Project Structure:$(NC)"
	@echo "  Source: $(SRC)/"
	@echo "  Tests: $(TESTS)/"
	@echo "  Docs: $(DOCS)/"
	@echo "  Examples: examples/"

# Quick shortcuts
t: test-fast ## Shortcut for test-fast
c: clean ## Shortcut for clean
f: format ## Shortcut for format
l: lint ## Shortcut for lint
u: up ## Shortcut for up
d: down ## Shortcut for down
h: help ## Shortcut for help
