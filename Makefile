# Makefile for CausalBoundingEngine

.PHONY: help docs test lint install clean

help:
	@echo "Available commands:"
	@echo "  install    Install package in development mode"
	@echo "  test       Run tests"
	@echo "  lint       Run code quality checks"
	@echo "  docs       Build documentation"
	@echo "  clean      Clean build artifacts"

install:
	pip install -e .[full,docs]

test:
	pytest tests/ -v

lint:
	black causalboundingengine/ tests/
	isort causalboundingengine/ tests/
	mypy causalboundingengine/

docs:
	cd docs && make html

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf docs/build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
