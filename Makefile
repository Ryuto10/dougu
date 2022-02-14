all: lint test

lint:
	poetry run isort .
	poetry run black .
	poetry run mypy dougu --show-error-context --disallow-untyped-defs --ignore-missing-imports --no-incremental
	poetry run mypy tests --show-error-context --disallow-untyped-defs --ignore-missing-imports --no-incremental

test:
	poetry run pytest -v --capture=no

build_docker:
	docker-compose -f docker/docker-compose.yaml up -d --build

run_docker:
	docker-compose -f docker/docker-compose.yaml run python-work bash
