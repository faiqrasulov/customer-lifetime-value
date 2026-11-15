PYTHON ?= python
MEMCAP ?=
DATA_HOME ?= data

# Pin BLAS/OpenMP threads: deterministic timings and no oversubscription on small machines.
export OMP_NUM_THREADS ?= 2
export OPENBLAS_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
export VECLIB_MAXIMUM_THREADS ?= 2

.PHONY: install data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Download the UCI Online Retail zip into $(DATA_HOME)/raw and cache the parsed parquet.
data:
	$(MEMCAP) $(PYTHON) -m clv.cli download --data-home $(DATA_HOME)

# Full reproduction: cleaning, holdout validation, full-data refit, CLV segments
# -> reports/metrics.json, reports/segments.csv, reports/figures/*.png
run: data
	$(MEMCAP) $(PYTHON) -m clv.cli run --data-home $(DATA_HOME) --out reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
