DEPS_dd_dtw = $(shell python3 generate.py -dq dd_dtw.c)
DEPS_dd_dtw_openmp := $(shell python3 generate.py -dq dd_dtw_openmp.c)
DEPS_dd_ed := $(shell python3 generate.py -dq dd_ed.c)
TARGETS = ${shell python3 generate.py -ta}


.PHONY: generate
generate: jinja replace

.PHONY: clean
clean:
	rm -f $(TARGETS)

.PHONY: jinja
jinja: dd_dtw.c dd_dtw_openmp.c dd_ed.c

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

dd_ed.c: $(DEPS_dd_ed)
	@echo "Changed:" $?
	python3 generate.py $@

../dd_ed.c: dd_ed.c
	cp dd_ed.c ../

.PHONY: replace
replace: ../dd_dtw.c ../dd_dtw_openmp.c ../dd_ed.c

.PHONY: debug
debug:
	@echo $(DEPS_dd_dtw)
	@echo $(DEPS_dd_dtw_openmp)

