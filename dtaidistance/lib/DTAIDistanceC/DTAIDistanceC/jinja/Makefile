DEPS_dd_dtw = $(shell python3 generate.py -dq dd_dtw.c)
DEPS_dd_dtw_openmp := $(shell python3 generate.py -dq dd_dtw_openmp.c)
DEPS_dd_ed := $(shell python3 generate.py -dq dd_ed.c)
DEPS_dd_loco := $(shell python3 generate.py -dq dd_loco.c)
TARGETS_ALL = ${shell python3 generate.py -ta}
TARGETS = ${shell python3 generate.py -t}


.PHONY: generate
generate: jinja replace

.PHONY: clean
clean:
	rm -f $(TARGETS_ALL)

.PHONY: jinja
jinja: $(TARGETS)

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

dd_loco.c: $(DEPS_dd_loco)
	@echo "Changed:" $?
	python3 generate.py $@

../dd_loco.c: dd_loco.c
	cp dd_loco.c ../

.PHONY: replace
replace: ../dd_dtw.c ../dd_dtw_openmp.c ../dd_ed.c ../dd_loco.c

.PHONY: debug
debug:
	@echo $(DEPS_dd_dtw)
	@echo $(DEPS_dd_dtw_openmp)

