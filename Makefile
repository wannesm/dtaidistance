
# BENCHMARKSETTINGS = --ignore=venv -vv --benchmark-autosave --benchmark-storage=file://./benchmark-results --benchmark-disable-gc --benchmark-histogram=./benchmark-results/benchmark_$(shell date +%Y%m%d_%H%M%S) --benchmark-only
BENCHMARKSETTINGS = --ignore=venv -vv --benchmark-autosave --benchmark-disable-gc --benchmark-histogram=./benchmark-results/benchmark_$(shell date +%Y%m%d_%H%M%S) --benchmark-only


.PHONY: default
default:
	@echo "Possible actions:"
	@echo "- analyze_build"
	@echo "- benchmark"
	@echo "- build"
	@echo "- clean"
	@echo "- test"
	@echo "- testall"

.PHONY: runtest
runtest:
	export PYTHONPATH=.;python3 tests/test_bugs.py

.PHONY: test
test:
	export PYTHONPATH=.;py.test --ignore=venv --benchmark-skip -vv

.PHONY: testall
testall:
	export PYTHONPATH=.;py.test --ignore=venv -vv

.PHONY: benchmark
benchmark:
	export PYTHONPATH=.;py.test ${BENCHMARKSETTINGS}

.PHONY: benchmark-parallelc
benchmark-parallelc:
	export PYTHONPATH=.;py.test -k 'matrix1 or distance1' ${BENCHMARKSETTINGS}

.PHONY: benchmark-distancec
benchmark-distancec:
	export PYTHONPATH=.;py.test -k 'distance1' ${BENCHMARKSETTINGS}

.PHONY: benchmark-matrixc
benchmark-matrixc:
	export PYTHONPATH=.;py.test -k 'matrix1 and _c' ${BENCHMARKSETTINGS}

.PHONY: benchmark-clustering
benchmark-clustering:
	export PYTHONPATH=.;py.test -k cluster ${BENCHMARKSETTINGS}


.PHONY: clean
clean:
	python3 setup.py clean
	rm -f dtaidistance/dtw_c.{c,html,so}
	rm -f dtaidistance/dtw_c.*.so
	rm -f dtaidistance/dtw_cc.{c,html}
	rm -f dtaidistance/dtw_cc.*.so
	rm -f dtaidistance/dtw_cc_*.{c,html}
	rm -f dtaidistance/dtw_cc_*.*.so
	rm -f dtaidistance/ed_cc.{c,html}
	rm -f dtaidistance/ed_cc.*.so
	rm -f dtaidistance/util_*_cc.{c,html}
	rm -f dtaidistance/util_*_cc.*.so
	rm -f dtaidistance/*.pyc
	rm -rf dtaidistance/__pycache__

.PHONY: build
build:
	python3 setup.py build_ext --inplace

.PHONY: analyze_build
analyze_build:
	cd dtaidistance;cython dtw_c.pyx -a
	open dtaidistance/dtw_c.html

.PHONY: dist
dist:
	rm -rf dist/*
	python3 setup.py sdist

.PHONY: prepare_dist
prepare_dist:
	rm -rf dist/*
	python3 setup.py sdist bdist_wheel

.PHONY: deploy
deploy: prepare_dist
	@echo "Check whether repo is clean"
	git diff-index --quiet HEAD
	@echo "Add tag"
	git tag "v$$(python3 setup.py --version)"
	git push --tags
	@echo "Start uploading"
	twine upload --repository dtaidistance dist/*

.PHONY: docs
docs:
	export PYTHONPATH=..; cd docs; make html
