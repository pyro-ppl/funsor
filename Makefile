lint: FORCE
	flake8

test: lint FORCE
	pytest -v test.py

FORCE:
