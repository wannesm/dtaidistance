DEPS_dtw_cc = $(shell python3 generate.py -dq dtw_cc.pyx)
DEPS_dtw_cc_omp = $(shell python3 generate.py -dq dtw_cc_omp.pyx)
DEPS_globals = $(shell python3 generate.py -dq dtaidistancec_globals.pxd)
DEPS_dtw_cc_pxd = $(shell python3 generate.py -dq dtw_cc.pxd)
DEPS_ed_cc = $(shell python3 generate.py -dq ed_cc.pyx)


.PHONY: generate
generate: jinja replace

.PHONY: jinja
jinja: dtw_cc.pyx dtw_cc_omp.pyx dtaidistancec_globals.pxd dtw_cc.pxd

dtw_cc.pyx: $(DEPS_dtw_cc)
	@echo "Changed:" $?
	python3 generate.py $@

../dtw_cc.pyx: dtw_cc.pyx
	cp dtw_cc.pyx ../

dtw_cc_omp.pyx: $(DEPS_dtw_cc_omp)
	@echo "Changed:" $?
	python3 generate.py $@

../dtw_cc_omp.pyx: dtw_cc_omp.pyx
	cp dtw_cc_omp.pyx ../

dtw_cc.pxd: $(DEPS_dtw_cc_pxd)
	@echo "Changed:" $?
	python3 generate.py $@

../dtw_cc.pxd: dtw_cc.pxd
	cp dtw_cc.pxd ../

dtaidistancec_globals.pxd: $(DEPS_globals)
	@echo "Changed:" $?
	python3 generate.py $@

../dtaidistancec_globals.pxd: dtaidistancec_globals.pxd
	cp dtaidistancec_globals.pxd ../

ed_cc.pyx: $(DEPS_ed_cc)
	@echo "Changed:" $?
	python3 generate.py $@

../ed_cc.pyx: ed_cc.pyx
	cp ed_cc.pyx ../

.PHONY: replace
replace: ../dtw_cc.pyx ../dtw_cc_omp.pyx ../dtaidistancec_globals.pxd  ../dtw_cc.pxd ../ed_cc.pyx

.PHONY: debug
debug:
	@echo $(DEPS_dtw_cc)
	@echo $(DEPS_globals)
	@echo $(DEPS_dtw_cc_pxd)

