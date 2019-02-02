all: test

install:
	pip install -e .[dev]

lint: FORCE
	# TODO(eb8680) fix lint errors in minipyro.py
	flake8 --exclude=examples/minipyro.py

test: lint FORCE
	pytest -v test
	python examples/normal_filter.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
