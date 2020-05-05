.PHONY: all install docs lint format test clean FORCE

all: docs test

install:
	pip install -e .[dev]

docs: FORCE
	mkdir -p docs/source/_static
	$(MAKE) -C docs html

lint: FORCE
	flake8 --per-file-ignores='funsor/distributions.py:F821'

license: FORCE
	python scripts/update_headers.py

format: FORCE
	isort -y

test: lint FORCE
	pytest -v -n auto test/*py
	FUNSOR_BACKEND=torch pytest -v -n auto test/
	FUNSOR_BACKEND=jax pytest -v test/test_tensor.py
	FUNSOR_BACKEND=jax pytest -v test/test_gaussian.py
	FUNSOR_BACKEND=jax pytest -v test/test_distributions.py
	FUNSOR_BACKEND=torch FUNSOR_DEBUG=1 pytest -v test/test_gaussian.py
	FUNSOR_BACKEND=torch FUNSOR_USE_TCO=1 pytest -v test/test_terms.py
	FUNSOR_BACKEND=torch FUNSOR_USE_TCO=1 pytest -v test/test_einsum.py
	FUNSOR_BACKEND=torch python examples/discrete_hmm.py -n 2
	FUNSOR_BACKEND=torch python examples/discrete_hmm.py -n 2 -t 50 --lazy
	FUNSOR_BACKEND=torch FUNSOR_USE_TCO=1 python examples/discrete_hmm.py -n 1 -t 50 --lazy
	FUNSOR_BACKEND=torch FUNSOR_USE_TCO=1 python examples/discrete_hmm.py -n 1 -t 500 --lazy
	FUNSOR_BACKEND=torch python examples/kalman_filter.py -n 2
	FUNSOR_BACKEND=torch python examples/kalman_filter.py -n 2 -t 50 --lazy
	FUNSOR_BACKEND=torch FUNSOR_USE_TCO=1 python examples/kalman_filter.py -n 1 -t 50 --lazy
	FUNSOR_BACKEND=torch FUNSOR_USE_TCO=1 python examples/kalman_filter.py -n 1 -t 500 --lazy
	FUNSOR_BACKEND=torch python examples/minipyro.py
	FUNSOR_BACKEND=torch python examples/minipyro.py --jit
	FUNSOR_BACKEND=torch python examples/slds.py -n 2 -t 50
	FUNSOR_BACKEND=torch python examples/pcfg.py --size 3
	FUNSOR_BACKEND=torch python examples/vae.py --smoke-test
	FUNSOR_BACKEND=torch python examples/eeg_slds.py --num-steps 2 --fon --test 
	FUNSOR_BACKEND=torch python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --smoke
	FUNSOR_BACKEND=torch python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --parallel --smoke
	FUNSOR_BACKEND=torch python examples/sensor.py --seed=0 --num-frames=2 -n 1
	@echo PASS

clean: FORCE
	git clean -dfx -e funsor-egg.info

FORCE:
