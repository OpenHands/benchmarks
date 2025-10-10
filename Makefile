SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Colors for output
ECHO := printf '%b\n'
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m
RESET := \033[0m

# Required uv version
REQUIRED_UV_VERSION := 0.8.13

.PHONY: build format lint clean help check-uv-version

# Default target
.DEFAULT_GOAL := help


check-uv-version:
	@$(ECHO) "$(YELLOW)Checking uv version...$(RESET)"
	@UV_VERSION=$$(uv --version | cut -d' ' -f2); \
	REQUIRED_VERSION=$(REQUIRED_UV_VERSION); \
	if [ "$$(printf '%s\n' "$$REQUIRED_VERSION" "$$UV_VERSION" | sort -V | head -n1)" != "$$REQUIRED_VERSION" ]; then \
		$(ECHO) "$(RED)Error: uv version $$UV_VERSION is less than required $$REQUIRED_VERSION$(RESET)"; \
		$(ECHO) "$(YELLOW)Please update uv with: uv self update$(RESET)"; \
		exit 1; \
	fi; \
	$(ECHO) "$(GREEN)uv version $$UV_VERSION meets requirements$(RESET)"

build: check-uv-version
	@$(ECHO) "$(CYAN)Setting up OpenHands V1 development environment...$(RESET)"
	@$(ECHO) "$(YELLOW)Installing dependencies with uv sync --dev...$(RESET)"
	@uv sync --dev
	@$(ECHO) "$(GREEN)Dependencies installed successfully.$(RESET)"
	@$(ECHO) "$(YELLOW)Setting up pre-commit hooks...$(RESET)"
	@uv run pre-commit install
	@$(ECHO) "$(GREEN)Pre-commit hooks installed successfully.$(RESET)"
	@$(ECHO) "$(GREEN)Build complete! Development environment is ready.$(RESET)"

format:
	@$(ECHO) "$(YELLOW)Formatting code with uv format...$(RESET)"
	@uv run ruff format
	@$(ECHO) "$(GREEN)Code formatted successfully.$(RESET)"

lint:
	@$(ECHO) "$(YELLOW)Linting code with ruff...$(RESET)"
	@uv run ruff check --fix
	@$(ECHO) "$(GREEN)Linting completed.$(RESET)"

clean:
	@$(ECHO) "$(YELLOW)Cleaning up cache files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache .mypy_cache 2>/dev/null || true
	@$(ECHO) "$(GREEN)Cache files cleaned.$(RESET)"
