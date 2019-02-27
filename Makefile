all: test

install:
	pip install -e .[dev]

lint: FORCE
	flake8

test: lint FORCE
	pytest -v test
	python examples/discrete_hmm.py -n 2
	@#python examples/kalman_filter.py --xfail-if-not-implemented
	@#python examples/ss_vae_delayed.py --xfail-if-not-implemented
	@#python examples/minipyro.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
