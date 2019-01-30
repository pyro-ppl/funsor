test: FORCE
	flake8
	pytest -v test.py

FORCE:
