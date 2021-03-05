.PHONY: all install docs lint format test clean FORCE

all: docs test

install:
	pip install -e .[dev]

docs: FORCE
	mkdir -p docs/source/_static
	$(MAKE) -C docs html

lint: FORCE
	flake8
	black --check .
	isort --check .
	python scripts/update_headers.py --check

license: FORCE
	python scripts/update_headers.py

format: license FORCE
	black .
	isort .

test: lint FORCE
ifeq (${FUNSOR_BACKEND}, torch)
	pytest -v -n auto test/
	FUNSOR_DEBUG=1 pytest -v test/test_gaussian.py
	FUNSOR_PROFILE=99 pytest -v test/test_einsum.py
	FUNSOR_USE_TCO=1 pytest -v test/test_terms.py
	FUNSOR_USE_TCO=1 pytest -v test/test_einsum.py
	python examples/discrete_hmm.py -n 2
	python examples/discrete_hmm.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 python examples/discrete_hmm.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 python examples/discrete_hmm.py -n 1 -t 500 --lazy
	python examples/forward_backward.py -t 3
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
	python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --smoke
	python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --parallel --smoke
	python examples/sensor.py --seed=0 --num-frames=2 -n 1
	@echo PASS
else ifeq (${FUNSOR_BACKEND}, jax)
	pytest -v -n auto --ignore=test/examples --ignore=test/pyro --ignore=test/pyroapi --ignore=test/test_distribution.py --ignore=test/test_distribution_generic.py
	pytest -v -n auto test/test_distribution.py
	pytest -v -n auto test/test_distribution_generic.py
	@echo PASS
else
	# default backend
	pytest -v -n auto --ignore=test/examples --ignore=test/pyro --ignore=test/pyroapi
	@echo PASS
endif

clean: FORCE
	git clean -dfx -e funsor-egg.info

FORCE:
