.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache

lint:
	flake8 .
	mypy . --exclude '(^|/)llm_sdk(/|$$)' --warn-return-any --warn-unused-ignores --follow-imports=skip  --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
