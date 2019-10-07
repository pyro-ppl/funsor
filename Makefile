.PHONY: all install docs lint format test clean FORCE

all: docs test

install:
	pip install -e .[dev]

docs: FORCE
	mkdir -p docs/source/_static
	$(MAKE) -C docs html

lint: FORCE
	flake8

format: FORCE
	isort -y

sensor: FORCE
	python examples/sensor.py --num-sensors 4 --frames 10,20,30 -n 80
	python examples/sensor.py --num-sensors 4 --frames 10,20,30 -n 80 --no-bias

test: lint FORCE
	pytest -v -n auto test/
	FUNSOR_DEBUG=1 pytest -v test/test_gaussian.py
	FUNSOR_USE_TCO=1 pytest -v test/test_terms.py
	FUNSOR_USE_TCO=1 pytest -v test/test_einsum.py
	FUNSOR_USE_TCO=1 pytest -v test/test_numpy.py
	python examples/discrete_hmm.py -n 2
	python examples/discrete_hmm.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 python examples/discrete_hmm.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 python examples/discrete_hmm.py -n 1 -t 500 --lazy
	python examples/kalman_filter.py -n 2
	python examples/kalman_filter.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 python examples/kalman_filter.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 python examples/kalman_filter.py -n 1 -t 500 --lazy
	python examples/minipyro.py
	python examples/minipyro.py --jit
	python examples/slds.py -n 2 -t 50
	python examples/pcfg.py --size 3
	python examples/vae.py --smoke-test
	python examples/eeg_slds.py --num-steps 2 --fon --test 
	@#python examples/ss_vae_delayed.py --xfail-if-not-implemented
	@echo PASS

clean: FORCE
	git clean -dfx -e funsor-egg.info

FORCE:
