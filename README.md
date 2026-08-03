# Decision Intelligence Logistics Engine

An end-to-end decision system for logistics planning that combines demand forecasting, stochastic simulation, and network optimization. 

The project is designed to showcase production-oriented applied science and engineering skills at the intersection of:

- Operations Research
- Machine Learning
- Data Engineering
- MLOps
- API-based deployment

The aim is to showcase my methodological research skills, within an 'offline' environment.

## Project Goal

Build a scalable logistics decision engine that can:

1. Generate or ingest historical shipment and demand data
2. Forecast future demand — independently per destination
3. Simulate uncertain logistics scenarios
4. Optimize origin-destination flows under capacity and cost constraints
5. Expose the full pipeline through an API

This repository reflects how real-world planning systems are built: not only with mathematical models, but also with robust data pipelines, modular software design, and deployable services.

---

## Latest Release: v1.1

Validated through:
- 216 automated tests
- Reproducibility checks
- Optimization consistency tests
- Experiment infrastructure verification

New in v1.1:
- Three-way train/validation/test split for per-destination model selection, removing the
  selection bias of evaluating on the same window used to pick the winning model
- Fixed a silent bug where `ETSForecaster`/`SARIMAXForecaster` returned in-sample fitted
  values instead of genuine out-of-sample forecasts
- A 2×2 factorial experiment suite (`experiments/`) that decomposes the value of better
  forecasting vs. better optimization against a naive SME baseline

See: [docs/reports/V1_0_FUNCTIONALITY_TEST_REPORT.md](docs/reports/V1_0_FUNCTIONALITY_TEST_REPORT.md) (v1.0 baseline report)

---

## Architecture

Three-layer pipeline: **Data → Forecasting → Optimization**.

- Per-destination local model architecture (each destination independently trained and selected)
- Multi-period min-cost flow optimizer with inventory tracking
- FastAPI serving layer wiring both engines

See [docs/architecture.md](docs/architecture.md) for the full system diagram, LP formulation, and component breakdown.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/<your-username>/decision-intelligence-logistics-engine.git
cd decision-intelligence-logistics-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### Analysis Environment (notebooks-related dependencies)
pip install -r requirements-analysis.txt

# Run the full pipeline demo
python scripts/example_end_to_end_pipeline.py

# Run tests
python -m pytest tests/ -v
```

### Running the 2×2 Factorial Experiments

The `experiments/` module runs the paper's forecasting-vs-optimization value decomposition
(naive vs. DILE forecasting × naive vs. DILE optimization) against a versioned synthetic
dataset:

```bash
# Run all four scenarios (B00, B01, B10, B11) and print a consolidated summary
PYTHONPATH=src python experiments/run_all.py

# Run a single scenario
PYTHONPATH=src python experiments/run_experiment.py experiments/configs/B11_dile_forecast_dile_opt.yaml
```

Each run saves `planning_metrics.json`, `realized_metrics.json`, `forecasts.parquet`,
`flows.parquet`, `inventory.parquet`, and `config.yaml` to `experiments/results/<name>/`.

---

## API

Start the server:

```bash
PYTHONPATH=src uvicorn api.app:app --reload
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/forecast` | Per-destination demand forecasting |
| `POST` | `/optimize` | Multi-period min-cost flow optimization |
| `POST` | `/plan` | Full pipeline: forecast → optimize in one call |

See [docs/api.md](docs/api.md) for endpoint details, request schemas, and example requests/responses.

---

## Testing

```bash
python -m pytest tests/ -v
# 216 passed
```

Key correctness properties verified:
- Data isolation between destinations
- Temporal split correctness (no future leakage), including the three-way
  train/validation/test split used for model selection
- Row-order independence
- Model selection minimality with tiebreaking
- Fault tolerance completeness
- Determinism across executions
- Pipeline protocol conformance

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11+ |
| DataFrames | Polars |
| Optimization | OR-Tools (GLOP, CBC) |
| Statistical Models | statsmodels (ETS, SARIMAX) |
| Metrics | scikit-learn |
| Parallelism | joblib |
| Numerics | NumPy |
| Visualization | Matplotlib |
| Configuration | PyYAML |
| Testing | pytest, Hypothesis (property-based testing) |
| API | FastAPI, Uvicorn |

---

## Planned Features

- [x] FastAPI endpoints for end-to-end execution (`/forecast`, `/optimize`, `/plan`)
- [ ] Stochastic simulation layer implementation (interface defined via `SimulationInterface`)
- [ ] MLflow experiment tracking
- [ ] Docker support
- [ ] ML model integration (LightGBM, XGBoost, Prophet)
- [ ] Hierarchical forecasting
- [ ] Performance benchmarking
- [ ] Visualization config support (show/save via YAML)

---

## Author

**Christian Piermarini**
Applied Scientist / Operations Research / Machine Learning
