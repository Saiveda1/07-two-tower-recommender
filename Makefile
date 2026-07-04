.PHONY: setup data run test bench screenshots all clean

PY ?= python
export PYTHONPATH := src
export MPLBACKEND := Agg

# Scale knobs (override on the CLI, e.g. `make run ROWS=20000000`)
ROWS   ?= 5000000
USERS  ?= 50000
ITEMS  ?= 5000
EPOCHS ?= 4

setup:                       ## install deps (offline env already has these)
	$(PY) -m pip install -r requirements.txt

data:                        ## stream an interaction log to Parquet
	$(PY) scripts/generate_data.py --rows $(ROWS) --users $(USERS) --items $(ITEMS) --out data/interactions.parquet

run:                         ## train two-tower + ranker, evaluate vs baselines
	$(PY) scripts/run.py --rows $(ROWS) --users $(USERS) --items $(ITEMS) --epochs $(EPOCHS)

test:                        ## run the pytest suite (behavioural assertions)
	$(PY) -m pytest -q

bench:                       ## streaming-scale benchmark (bounded memory)
	$(PY) benchmarks/benchmark_scale.py --rows 1000000 10000000 100000000

screenshots:                 ## render PNGs from the last run into assets/
	$(PY) scripts/make_screenshots.py

all: run screenshots test

clean:
	rm -rf data/*.parquet data/*.npz assets/*.png benchmarks/scale_results.csv
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
