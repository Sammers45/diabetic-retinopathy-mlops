# Linux/macOS/CI task runner. On Windows without make, use: .\scripts\tasks.ps1 <target>
PY ?= python
PROFILE ?=
SET ?=
export PYTHONPATH := src
export MLFLOW_DISABLE_AGENT_HINT := 1

CFG := $(if $(PROFILE),--profile $(PROFILE),) $(foreach s,$(SET),--set $(s))

.PHONY: setup fixtures download data train train-baseline ui server test clean

setup:            ## Create .venv and install requirements
	$(PY) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

fixtures:         ## Generate SYNTHETIC test images into tests/fixtures/synthetic
	$(PY) scripts/make_synthetic_fixtures.py --out tests/fixtures/synthetic

download:         ## Phase 1: fetch the Kaggle data into data/raw
	$(PY) -m dr.download $(CFG)

data:             ## Phase 2-3: ingest, validate, split, preprocess
	$(PY) -m dr.data $(CFG)

train:            ## Phase 4-5: train EfficientNet, evaluate, register
	$(PY) -m dr.train $(CFG)

train-baseline:   ## Phase 4-5: train the from-scratch baseline CNN
	$(PY) -m dr.train $(CFG) --set model.family=baseline_cnn --set model.pretrained=false

ui:               ## MLflow UI on the local sqlite store
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

server:           ## MLflow tracking server (sqlite + served artifacts), no Docker needed
	$(PY) -m mlflow server --backend-store-uri sqlite:///mlruns.db \
		--artifacts-destination ./mlartifacts --serve-artifacts --host 127.0.0.1 --port 5000

test:
	$(PY) -m pytest -q

clean:            ## Remove caches and generated fixtures (keeps data/ and MLflow stores)
	rm -rf .pytest_cache tests/fixtures/synthetic
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
