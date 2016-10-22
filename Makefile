
default:
	@echo "Possible actions:"
	@echo "- test"

test:
	export PYTHONPATH=.;py.test --ignore=venv --benchmark-skip -vv

benchmark:
	export PYTHONPATH=.;py.test --ignore=venv -vv --benchmark-autosave --benchmark-disable-gc

build:
	cd dtaidistance;python3 setup.py build_ext --inplace

analyze_build:
	cd dtaidistance;cython dtw_c.pyx -a
	open dtaidistance/dtw_c.html

