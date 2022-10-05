DEPS_dd_dtw = $(shell python3 generate.py -dq dd_dtw.c)
DEPS_dd_dtw_openmp := $(shell python3 generate.py -dq dd_dtw_openmp.c)


.PHONY: generate
generate: jinja replace

.PHONY: jinja
jinja: dd_dtw.c dd_dtw_openmp.c

dd_dtw.c: $(DEPS_dd_dtw)
	@echo "Changed:" $?
	python3 generate.py $@

../dd_dtw.c: dd_dtw.c
	cp dd_dtw.c ../

dd_dtw_openmp.c: $(DEPS_dd_dtw_openmp)
	@echo "Changed:" $?
	python3 generate.py $@

../dd_dtw_openmp.c: dd_dtw_openmp.c
	cp dd_dtw_openmp.c ../

.PHONY: replace
replace: ../dd_dtw.c ../dd_dtw_openmp.c

.PHONY: debug
debug:
	@echo $(DEPS_dd_dtw)
	@echo $(DEPS_dd_dtw_openmp)

