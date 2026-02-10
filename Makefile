
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

.PHONY: pypy-test
pypy-test:
	export PYTHONPATH=.;pypy3 -m py.test --ignore=venv --benchmark-skip -vv -c pytest-nolibs.ini

.PHONY: test-windows
test-windows:
	pytest --ignore=venv --benchmark-skip -vv -c pytest-nolibs.ini

.PHONY: test-nolibs
test-nolibs:
	export PYTHONPATH=.;pytest --ignore=venv --benchmark-skip -vv -c pytest-nolibs.ini

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

.PHONY: benchmark-subseqsearch
benchmark-subseqsearch:
	export PYTHONPATH=.;py.test -k test_dtw_subseqsearch_eeg_lb ${BENCHMARKSETTINGS}


.PHONY: clean
clean:
	python3 setup.py clean
	rm -f src/dtaidistance/dtw_c.c
	rm -f src/dtaidistance/dtw_c.html
	rm -f src/dtaidistance/dtw_c.so
	rm -f src/dtaidistance/dtw_c.*.so
	rm -f src/dtaidistance/dtw_cc.c
	rm -f src/dtaidistance/dtw_cc.html
	rm -f src/dtaidistance/dtw_cc.*.so
	rm -f src/dtaidistance/dtw_cc_*.c
	rm -f src/dtaidistance/dtw_cc_*.html
	rm -f src/dtaidistance/dtw_cc_*.*.so
	rm -f src/dtaidistance/ed_cc.c
	rm -f src/dtaidistance/ed_cc.html
	rm -f src/dtaidistance/ed_cc.*.so
	rm -f src/dtaidistance/util_*_cc.c
	rm -f src/dtaidistance/util_*_cc.html
	rm -f src/dtaidistance/util_*_cc.*.so
	rm -f src/dtaidistance/*.pyc
	rm -rf src/dtaidistance/__pycache__

.PHONY: use-venv
use-venv:
	$(eval $@_TMP := $(shell python3 -c 'import sys; print(sys.prefix)'))
	@#@echo $($@_TMP)
	@if [ -f "use_venv.txt" ]; then grep '$($@_TMP)' use_venv.txt || (echo "venv does not appear in use_venv.txt: $($@_TMP)"; exit 1) ;fi

.PHONY: build
build: use-venv
	python3 setup.py build_ext --inplace

.PHONY: pypy-build
pypy-build:
	pypy3 setup.py build_ext --inplace

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

.PHONY: add_to_dist
add_to_dist: build test
	$(eval $@_TMP := $(shell python3 -c 'import sys; print(sys.prefix)'))
	@echo "==============================================================="
	@echo "Using virtual env: $(@_TMP)"
	@echo "==============================================================="
	python3 setup.py sdist bdist_wheel

.PHONY: prepare_tag
prepare_tag:
	@echo "Check whether repo is clean"
	git diff-index --quiet HEAD
	@echo "Check correct branch"
	if [[ "$$(git rev-parse --abbrev-ref HEAD)" != "deploy" ]]; then echo 'Not deploy branch'; exit 1; fi
	@echo "Add tag"
	git tag "v$$(python3 setup.py --version)"
	git push --tags

.PHONY: deploy
deploy: prepare_dist prepare_tag
	@echo "Check whether repo is clean"
	git diff-index --quiet HEAD
	@echo "Manual action: Push to deploy Github branch to deploy"
	#@echo "Start uploading"
	#twine upload --repository dtaidistance dist/*

.PHONY: upload_from_local_machine
upload_from_local_machine:
	@echo "Did you run 'make add_to_dist' first?"
	@echo "Uploading the following distributions:"
	@ls dist/
	@echo "Are you sure? [y/n] " && read ans && ( [ $${ans:-N} = y ] || ( echo "Aborted" && exit 1 ) )
	@echo "Start uploading"
	twine upload --repository dtaidistance dist/*


.PHONY: docs
docs:
	export PYTHONPATH=..; cd docs; make html

