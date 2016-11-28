
default:
	@echo "Possible actions:"
	@echo "- analyze_build"
	@echo "- benchmark"
	@echo "- clean"
	@echo "- build"
	@echo "- test"

test:
	export PYTHONPATH=.;py.test --ignore=venv --benchmark-skip -vv

benchmark:
	export PYTHONPATH=.;py.test --ignore=venv -vv --benchmark-autosave --benchmark-disable-gc --benchmark-histogram

clean:
	rm -f dtaidistance/dtw_c.c
	rm -f dtaidistance/dtw_c.*.so

build:
	cd dtaidistance;python3 setup.py build_ext --inplace

analyze_build:
	cd dtaidistance;cython dtw_c.pyx -a
	open dtaidistance/dtw_c.html

