PYTHON ?= poetry run python

.PHONY: test demo-snapshot daily-update report

test:
	poetry run pytest -q

demo-snapshot:
	$(PYTHON) scripts/demo_snapshot_pipeline.py

daily-update:
	$(PYTHON) scripts/run_canonical_daily_update.py

report:
	$(PYTHON) scripts/generate_report.py --output-dir output/canonical_daily
