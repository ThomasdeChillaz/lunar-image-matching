.PHONY: test test-archive smoke audit replay check
PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

test-archive:
	cd archive/research && $(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m lunar_matching smoke

audit:
	$(PYTHON) -m lunar_matching audit

replay:
	$(PYTHON) -m lunar_matching replay

check: test test-archive audit smoke replay
