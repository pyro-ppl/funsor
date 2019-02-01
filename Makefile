all: test

lint: FORCE
	flake8

test: lint FORCE
	pytest -v test.py
	python example.py --xfail-if-not-implemented
	@echo PASS

FORCE:
