.PHONY: test smoke fetch evaluate review
PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) benchmark.py --config smoke.json --output runs/smoke-new

fetch:
	$(PYTHON) dataset.py fetch
	$(PYTHON) fetch_models.py

evaluate:
	$(PYTHON) reproduce.py --output runs/reproduction

review:
	$(PYTHON) checkpoint_review.py --output runs/checkpoint_review.html
