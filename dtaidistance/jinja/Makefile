DEPS_dtw_cc = $(shell python3 generate.py -dq dtw_cc.pyx)


.PHONY: generate
generate: jinja replace

.PHONY: jinja
jinja: dtw_cc.pyx

dtw_cc.pyx: $(DEPS_dtw_cc)
	@echo "Changed:" $?
	python3 generate.py $@

../dtw_cc.pyx: dtw_cc.pyx
	cp dtw_cc.pyx ../

.PHONY: replace
replace: ../dtw_cc.pyx

.PHONY: debug
debug:
	@echo $(DEPS_dtw_cc)


