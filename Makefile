all: test

install:
	pip install -e .[dev]

lint: FORCE
	flake8

test: lint FORCE
	pytest -v test
	python examples/normal_filter.py --xfail-if-not-implemented
	python examples/minipyro.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
