all: test

install:
	pip install -e .[dev]

lint: FORCE
	flake8

test: lint FORCE
	pytest -v test
	python examples/normal_filter.py --xfail-if-not-implemented
	@echo 'TODO(eb8680) fix examples/minipyro.py'
	#python examples/minipyro.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
