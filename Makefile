.PHONY: clean-pyc dependencies

.DEFAULT: help

help:
	@echo "make clean-pyc"
	@echo "       Remove generated .pyc files."
	@echo "make dependencies"
	@echo "       Install required enviroment dependences and packages."
	@echo "make test"
	@echo "       run application tests."
	@echo "make run"
	@echo "       run application without arguments."
	@echo "       [OPTIONAL]: If running application with arguments, run with 'python -m adaboost {arguments}'"
	@echo "                   as described below"
	@echo ""

	python -m adaboost help

clean-pyc:
	find . -name '*.pyc' -delete

dependencies:
	pip install -r requirements.txt

test:
	python -m unittest discover adaboost/test -v -b

run:
	python -m adaboost
