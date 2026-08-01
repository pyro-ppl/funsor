.PHONY: all install docs lint format test clean FORCE

all: docs test

install:
	uv sync

docs: FORCE
	uv sync --group docs --no-default-groups --inexact
	mkdir -p docs/source/_static
	$(MAKE) -C docs html SPHINXBUILD="uv run --no-sync sphinx-build"

lint: FORCE
	uv run ruff check --fix .
	uv run python scripts/update_headers.py --check
	uv run python test/test_import.py

license: FORCE
	uv run python scripts/update_headers.py

format: license FORCE
	uv run ruff format .

test: lint FORCE
ifeq (${FUNSOR_BACKEND}, torch)
	uv sync --extra torch --inexact
	uv run pytest -v -n auto test/
	FUNSOR_DEBUG=1 uv run pytest -v test/test_gaussian.py
	FUNSOR_PROFILE=99 uv run pytest -v test/test_einsum.py
	FUNSOR_USE_TCO=1 uv run pytest -v test/test_terms.py
	FUNSOR_USE_TCO=1 uv run pytest -v test/test_einsum.py
	uv run python examples/adam.py -n 2
	uv run python examples/discrete_hmm.py -n 2
	uv run python examples/discrete_hmm.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 uv run python examples/discrete_hmm.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 uv run python examples/discrete_hmm.py -n 1 -t 500 --lazy
	uv run python examples/forward_backward.py -t 3
	uv run python examples/kalman_filter.py -n 2
	uv run python examples/kalman_filter.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 uv run python examples/kalman_filter.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 uv run python examples/kalman_filter.py -n 1 -t 500 --lazy
	uv run python examples/minipyro.py
	uv run python examples/minipyro.py --jit
	uv run python examples/slds.py -n 2 -t 50
	uv run python examples/pcfg.py --size 3
	uv run python examples/talbot.py -n 2
	uv run python examples/vae.py --smoke-test
	uv run python examples/eeg_slds.py --num-steps 2 --fon --test
	uv run python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --smoke
	uv run python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --parallel --smoke
	uv run python examples/sensor.py --seed=0 --num-frames=2 -n 1
	uv run python examples/adam.py --num-steps=21
	@echo PASS
else ifeq (${FUNSOR_BACKEND}, jax)
	uv sync --extra jax --inexact
	uv run pytest -v -n auto --ignore=test/examples --ignore=test/pyro --ignore=test/pyroapi \
		--ignore=test/test_distribution.py --ignore=test/test_distribution_generic.py \
		--ignore=test/torch
	uv run pytest -v -n auto test/test_distribution.py
	uv run pytest -v -n auto test/test_distribution_generic.py
	@echo PASS
else
	# default backend; requires prior `make install` (or equivalent uv sync)
	uv run pytest -v -n auto --ignore=test/examples --ignore=test/pyro \
		--ignore=test/pyroapi --ignore=test/torch
	@echo PASS
endif

clean: FORCE
	git clean -dfx -e funsor-egg.info

FORCE:
