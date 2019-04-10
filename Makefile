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
	export PYTHONPATH=.;py.test --ignore=venv -vv --benchmark-autosave --benchmark-disable-gc --benchmark-histogram --benchmark-only

.PHONY: benchmark-parallelc
benchmark-parallelc:
	export PYTHONPATH=.;py.test -k 'matrix1 and pure' --ignore=venv -vv --benchmark-autosave --benchmark-disable-gc --benchmark-histogram --benchmark-only

.PHONY: benchmark-clustering
benchmark-clustering:
	export PYTHONPATH=.;py.test -k cluster --ignore=venv -vv --benchmark-autosave --benchmark-disable-gc --benchmark-histogram --benchmark-only


.PHONY: clean
clean:
	python3 setup.py clean
	rm -f dtaidistance/dtw_c.c
	rm -f dtaidistance/dtw_c.*.so

.PHONY: build
build:
	python3 setup.py build_ext --inplace

.PHONY: analyze_build
analyze_build:
	cd dtaidistance;cython dtw_c.pyx -a
	open dtaidistance/dtw_c.html

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
	@echo "Start uploading"
	twine upload dist/*

.PHONY: docs
docs:
	export PYTHONPATH=..; cd docs; make html
