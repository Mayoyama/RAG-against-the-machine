ARGS =

.PHONY: run install debug lint lint-strict clean uninstall

run:
	uv run python -m src $(ARGS)

install:
	uv sync

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run python -m flake8 . --exclude=.venv/
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=.venv/

lint-strict:
	uv run python -m flake8 . --exclude=.venv/
	uv run python -m mypy . --strict --exclude=.venv/

clean:
	find . -not -path './.venv/*' -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \)  -exec rm -rf {} +
	find . -type f \( -name "*.pyo" -o -name "*.pyc" -o -name "*~" \) -delete

uninstall:
	rm -rf .venv