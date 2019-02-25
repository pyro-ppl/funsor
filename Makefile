all: test

install:
	pip install -e .[dev]

lint: FORCE
	flake8

test: lint FORCE
	pytest -v test
	@#python examples/discrete_hmm.py --xfail-if-not-implemented
	@#python examples/kalman_filter.py --xfail-if-not-implemented
	@#python examples/ss_vae_delayed.py --xfail-if-not-implemented
	@#python examples/minipyro.py --xfail-if-not-implemented
	@echo PASS

test-cuda:
	TORCH_TENSOR_TYPE=torch.cuda.FloatTensor pytest -v

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
