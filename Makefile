
default:
	@echo "Possible actions:"
	@echo "- test"

test:
	export PYTHONPATH=.;py.test --ignore=venv -vv

build:
	cd dtaidistance;python3 setup.py build_ext --inplace
