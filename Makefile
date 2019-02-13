all: test

install:
	pip install -e .[dev]

lint: FORCE
	flake8 --ignore=F811,E121,E123,E126,E226,E24,E704,W503,W504

test: lint FORCE
	pytest -v test
	python examples/normal_filter.py --xfail-if-not-implemented
	@echo 'TODO(eb8680) fix examples/minipyro.py'
	#python examples/minipyro.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
