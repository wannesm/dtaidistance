DEPS_dtw_cc = $(shell python3 generate.py -dq dtw_cc.pyx)
DEPS_dtw_cc_omp = $(shell python3 generate.py -dq dtw_cc_omp.pyx)
DEPS_globals = $(shell python3 generate.py -dq dtaidistancec_globals.pxd)
DEPS_dtw_cc_pxd = $(shell python3 generate.py -dq dtw_cc.pxd)
DEPS_ed_cc = $(shell python3 generate.py -dq ed_cc.pyx)
ARGS := 

# ifeq ($(OS),Windows_NT) 
#     detected_OS := Windows
# else
#     detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
# endif
# SED = sed
# ifeq ($(detected_OS),Darwin)
#     SED = sed -i
# endif


.PHONY: default
default: clean generate

.PHONY: double
double: default

.PHONY: float
float: ARGS := --seqt=float
float: clean generate
	@# $(SED) '' 's/^typedef .* seq_t;$$/typedef float seq_t;/g' ../lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h

.PHONY: int
int: ARGS := --seqt=int
int: clean generate
	@# $(SED) '' 's/^typedef .* seq_t;$$/typedef int seq_t;/g' ../lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h

.PHONY: generate
generate: jinja replace
	@# $(SED) '' 's/^typedef .* seq_t;$$/typedef double seq_t;/g' ../lib/DTAIDistanceC/DTAIDistanceC/dd_globals.h

.PHONY: jinja
jinja: dtw_cc.pyx dtw_cc_omp.pyx dtaidistancec_globals.pxd dtw_cc.pxd
	python3 generate.py $(ARGS) dd_globals.h

dtw_cc.pyx: $(DEPS_dtw_cc)
	@echo "Changed:" $?
	python3 generate.py $(ARGS) $@

../dtw_cc.pyx: dtw_cc.pyx
	cp dtw_cc.pyx ../

dtw_cc_omp.pyx: $(DEPS_dtw_cc_omp)
	@echo "Changed:" $?
	python3 generate.py $(ARGS) $@

../dtw_cc_omp.pyx: dtw_cc_omp.pyx
	cp dtw_cc_omp.pyx ../

dtw_cc.pxd: $(DEPS_dtw_cc_pxd)
	@echo "Changed:" $?
	python3 generate.py $(ARGS) $@

../dtw_cc.pxd: dtw_cc.pxd
	cp dtw_cc.pxd ../

dtaidistancec_globals.pxd: $(DEPS_globals)
	@echo "Changed:" $?
	python3 generate.py $(ARGS) $@

../dtaidistancec_globals.pxd: dtaidistancec_globals.pxd
	cp dtaidistancec_globals.pxd ../

ed_cc.pyx: $(DEPS_ed_cc)
	@echo "Changed:" $?
	python3 generate.py $(ARGS) $@

../ed_cc.pyx: ed_cc.pyx
	cp ed_cc.pyx ../

.PHONY: replace
replace: ../dtw_cc.pyx ../dtw_cc_omp.pyx ../dtaidistancec_globals.pxd  ../dtw_cc.pxd ../ed_cc.pyx

.PHONY: debug
debug:
	@echo $(DEPS_dtw_cc)
	@echo $(DEPS_globals)
	@echo $(DEPS_dtw_cc_pxd)

.PHONY: clean
clean:
	rm -f dtw_cc.pyx
	rm -f dtw_cc_omp.pyx
	rm -f dtw_cc.pxd
	rm -f ed_cc.pyx
	rm -f dtaidistancec_globals.pxd

