ARGS =

.PHONY: run api install index index-bonus debug lint lint-strict clean uninstall

run:
	uv run python -m src $(ARGS)

api:
	uv run uvicorn src.api:app --reload

install:
	uv sync

setup-data:
	test -f vllm-0.10.1.zip || (echo "setup-data: missing vllm-0.10.1.zip in repo root" && exit 1)
	test -f datasets_public.zip || (echo "setup-data: missing datasets_public.zip in repo root" && exit 1)
	test -f moulinette.zip || (echo "setup-data: missing moulinette.zip in repo root" && exit 1)
	mkdir -p data/raw
	mkdir -p data/datasets
	unzip -o vllm-0.10.1.zip -d data/raw
	unzip -o datasets_public.zip -d .
	rm -rf data/datasets/AnsweredQuestions
	rm -rf data/datasets/UnansweredQuestions
	mv datasets_public/public/AnsweredQuestions data/datasets/
	mv datasets_public/public/UnansweredQuestions data/datasets/
	rm -rf datasets_public
	unzip -o moulinette.zip moulinette-ubuntu -d .
	rm -f moulinette
	mv moulinette-ubuntu moulinette
	chmod +x moulinette

index: 
	uv run python -m src index --root_directory data/raw --save_directory data/processed

index-bonus: 
	uv run python -m src index --root_directory data/raw --save_directory data/processed --apply_bonuses True

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run python -m flake8 . --exclude=.venv/,data/
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude=.venv/ --exclude=data/

lint-strict:
	uv run python -m flake8 . --exclude=.venv/,data/
	uv run python -m mypy . --strict --exclude=.venv/ --exclude=data/

clean:
	find . -not -path './.venv/*' -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \)  -exec rm -rf {} +
	find . -type f \( -name "*.pyo" -o -name "*.pyc" -o -name "*~" \) -delete

uninstall:
	rm -rf .venv
	rm -rf data
