.PHONY: all install docs lint format test clean FORCE

# After each target's explicit `uv sync --inexact`, avoid `uv run` pulling default groups.
UV_RUN = uv run --no-sync

all: docs test

install:
	uv sync --group dev --no-default-groups --inexact

docs: FORCE
	uv sync --group docs --no-default-groups --inexact
	mkdir -p docs/source/_static
	$(MAKE) -C docs html SPHINXBUILD="$(UV_RUN) sphinx-build"

lint: FORCE
	uv sync --group test --no-default-groups --inexact
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) python scripts/update_headers.py --check
	$(UV_RUN) python test/test_import.py

license: FORCE
	uv sync --group test --no-default-groups --inexact
	$(UV_RUN) python scripts/update_headers.py

format: license FORCE
	uv sync --group test --no-default-groups --inexact
	$(UV_RUN) ruff format .

test: lint FORCE
ifeq (${FUNSOR_BACKEND}, torch)
	uv sync --group test --extra torch --no-default-groups --inexact
	$(UV_RUN) pytest -v -n auto test/
	FUNSOR_DEBUG=1 $(UV_RUN) pytest -v test/test_gaussian.py
	FUNSOR_PROFILE=99 $(UV_RUN) pytest -v test/test_einsum.py
	FUNSOR_USE_TCO=1 $(UV_RUN) pytest -v test/test_terms.py
	FUNSOR_USE_TCO=1 $(UV_RUN) pytest -v test/test_einsum.py
	$(UV_RUN) python examples/adam.py -n 2
	$(UV_RUN) python examples/discrete_hmm.py -n 2
	$(UV_RUN) python examples/discrete_hmm.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 $(UV_RUN) python examples/discrete_hmm.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 $(UV_RUN) python examples/discrete_hmm.py -n 1 -t 500 --lazy
	$(UV_RUN) python examples/forward_backward.py -t 3
	$(UV_RUN) python examples/kalman_filter.py -n 2
	$(UV_RUN) python examples/kalman_filter.py -n 2 -t 50 --lazy
	FUNSOR_USE_TCO=1 $(UV_RUN) python examples/kalman_filter.py -n 1 -t 50 --lazy
	FUNSOR_USE_TCO=1 $(UV_RUN) python examples/kalman_filter.py -n 1 -t 500 --lazy
	$(UV_RUN) python examples/minipyro.py
	$(UV_RUN) python examples/minipyro.py --jit
	$(UV_RUN) python examples/slds.py -n 2 -t 50
	$(UV_RUN) python examples/pcfg.py --size 3
	$(UV_RUN) python examples/talbot.py -n 2
	$(UV_RUN) python examples/vae.py --smoke-test
	$(UV_RUN) python examples/eeg_slds.py --num-steps 2 --fon --test
	$(UV_RUN) python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --smoke
	$(UV_RUN) python examples/mixed_hmm/experiment.py -d seal -i discrete -g discrete -zi --parallel --smoke
	$(UV_RUN) python examples/sensor.py --seed=0 --num-frames=2 -n 1
	$(UV_RUN) python examples/adam.py --num-steps=21
	@echo PASS
else ifeq (${FUNSOR_BACKEND}, jax)
	uv sync --group test --extra jax --no-default-groups --inexact
	$(UV_RUN) pytest -v -n auto --ignore=test/examples --ignore=test/pyro --ignore=test/pyroapi \
		--ignore=test/test_distribution.py --ignore=test/test_distribution_generic.py \
		--ignore=test/torch
	$(UV_RUN) pytest -v -n auto test/test_distribution.py
	$(UV_RUN) pytest -v -n auto test/test_distribution_generic.py
	@echo PASS
else
	# default backend; lint already synced the test group
	$(UV_RUN) pytest -v -n auto --ignore=test/examples --ignore=test/pyro \
		--ignore=test/pyroapi --ignore=test/torch
	@echo PASS
endif

clean: FORCE
	git clean -dfx -e funsor-egg.info

FORCE:
