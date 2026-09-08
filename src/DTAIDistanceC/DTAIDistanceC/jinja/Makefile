DEPS_dd_dtw = $(shell ./generate.py -dq dd_dtw.c)
DEPS_dd_dtw_openmp := $(shell ./generate.py -dq dd_dtw_openmp.c)
DEPS_dd_ed := $(shell ./generate.py -dq dd_ed.c)
DEPS_dd_loco := $(shell ./generate.py -dq dd_loco.c)
TARGETS_ALL = ${shell ./generate.py -ta}
TARGETS = ${shell ./generate.py -t}


.PHONY: generate
generate: jinja replace

.PHONY: clean
clean:
	rm -f $(TARGETS_ALL)

.PHONY: jinja
jinja: $(TARGETS)

dd_dtw.c: $(DEPS_dd_dtw)
	@echo "Changed:" $?
	./generate.py $@

../dd_dtw.c: dd_dtw.c
	cp dd_dtw.c ../

dd_dtw_openmp.c: $(DEPS_dd_dtw_openmp)
	@echo "Changed:" $?
	./generate.py $@

../dd_dtw_openmp.c: dd_dtw_openmp.c
	cp dd_dtw_openmp.c ../

dd_ed.c: $(DEPS_dd_ed)
	@echo "Changed:" $?
	./generate.py $@

../dd_ed.c: dd_ed.c
	cp dd_ed.c ../

dd_loco.c: $(DEPS_dd_loco)
	@echo "Changed:" $?
	./generate.py $@

../dd_loco.c: dd_loco.c
	cp dd_loco.c ../

.PHONY: replace
replace: ../dd_dtw.c ../dd_dtw_openmp.c ../dd_ed.c ../dd_loco.c

.PHONY: debug
debug:
	@echo $(DEPS_dd_dtw)
	@echo $(DEPS_dd_dtw_openmp)

