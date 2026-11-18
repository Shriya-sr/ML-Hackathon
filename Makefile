# Convenience targets (GNU make). Windows users: run the commands shown in README.md directly.
PY ?= .venv/Scripts/python.exe

.PHONY: install eda features train optimise hmm evaluate heldout maps analysis report test api ui all

install:
	python -m venv .venv && $(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .
	cd frontend && npm ci

eda:        ; $(PY) scripts/run_eda.py
features:   ; $(PY) scripts/build_features.py --jobs 6
train:      ; $(PY) -W ignore scripts/train_all.py --fresh
optimise:   ; $(PY) -W ignore scripts/optimise.py
hmm:        ; $(PY) scripts/tune_hmm.py
evaluate:   ; $(PY) scripts/evaluate_all.py
heldout:    ; $(PY) scripts/evaluate_heldout.py   # final held-out score, run once
maps:       ; $(PY) scripts/generate_maps.py
analysis:   ; $(PY) scripts/feature_analysis.py
report:     ; $(PY) scripts/make_report_figures.py && $(PY) scripts/make_explainer_figures.py && $(PY) scripts/build_report.py
test:       ; $(PY) -m pytest -p no:warnings && cd frontend && npx vitest run
api:        ; $(PY) -m backend.main
ui:         ; cd frontend && npm run dev

all: features train optimise hmm evaluate analysis maps report
