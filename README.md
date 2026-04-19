# Decision Intelligence Logistics Engine

An end-to-end decision system for logistics planning that combines demand forecasting, stochastic simulation, and network optimization.

The project is designed to showcase production-oriented applied science and engineering skills at the intersection of:

- Operations Research
- Machine Learning
- Data Engineering
- MLOps
- API-based deployment

## Project Goal

The objective of this project is to build a scalable logistics decision engine that can:

1. generate or ingest historical shipment and demand data,
2. forecast future demand,
3. simulate uncertain logistics scenarios,
4. optimize origin-destination flows under capacity and cost constraints,
5. expose the full pipeline through an API.

This repository is meant to reflect how real-world planning systems are built: not only with mathematical models, but also with robust data pipelines, modular software design, and deployable services.

## Core Components

### 1. Data Layer
- Synthetic or open logistics data generation
- Data processing with Polars
- Analytical queries with DuckDB
- Efficient storage in Parquet format

### 2. Forecasting Layer
- Demand prediction using baseline models (Naive, Seasonal Lag, Rolling Window)
- Automatic model evaluation with MAE, MSE, MAPE, WAPE metrics
- Model selection: automatic ranking and best-model selection by configurable metric
- Forecast extraction into standardized schema and demand aggregation per destination
- Feature engineering with lag and rolling statistics
- Experiment tracking with MLflow

### 3. Simulation Layer
- Event-driven simulation of shipment arrivals, delays, and processing
- Stochastic demand generation
- Scenario analysis under uncertainty

### 4. Optimization Layer
- Minimum-cost transportation LP using OR-Tools (GLOP / CBC solvers)
- Capacity-constrained origin-to-destination flow assignment
- Input validation with pre-solve feasibility checks (unreachable destinations, insufficient capacity)
- Integration of forecast-derived demand into downstream optimization

### 5. Serving Layer
- FastAPI endpoints for simulation, forecasting, and optimization
- Reproducible configuration and modular architecture

## Tech Stack

- Python 3.11+
- Polars — high-performance DataFrames
- OR-Tools — linear programming solvers (GLOP, CBC)
- Scikit-learn — forecasting evaluation metrics
- NumPy — numerical operations
- Matplotlib — visualization
- PyYAML — configuration management
- pytest / Hypothesis — testing and property-based testing
- DuckDB (planned)
- FastAPI (planned)
- MLflow (planned)
- SimPy (planned, for simulation extensions)

## Repository Structure

```text
decision-intelligence-logistics-engine/
│
├── data/
│   ├── synthetic/          # parquet files (demand_history, origins, destinations, lanes)
│   └── output/             # metrics summaries, plots
├── notebooks/              # exploratory analysis and prototyping
├── src/
│   ├── data/
│   │   ├── ingestion.py        # Reader: parquet file loading
│   │   ├── input_data.py       # InputData dataclass
│   │   └── processing/         # per-dataset processors (demand, origins, lanes, destinations)
│   ├── forecasting/
│   │   ├── models/             # BaseForecaster, NaiveForecaster, SeasonalForecaster, RollingWindowForecaster
│   │   ├── pipeline.py         # ForecastingPipeline: runs models sequentially
│   │   ├── evaluator.py        # Evaluator: MAE, MSE, MAPE, WAPE
│   │   ├── model_selector.py   # ModelSelector: best-model selection and ranking
│   │   └── forecast_extractor.py  # ForecastExtractor: extraction and demand aggregation
│   ├── optimization/
│   │   └── optimizer.py        # Optimizer: min-cost transportation LP (OR-Tools)
│   ├── postprocessing/
│   │   ├── metrics_summary.py  # MetricsSummary: collect and export evaluation results
│   │   └── visualization.py    # VisualizationEngine: time series plots
│   ├── simulation/             # (planned) stochastic and event-driven simulation
│   ├── api/                    # (planned) FastAPI endpoints
│   └── utils/                  # config loading, system paths
│
├── tests/                  # unit and integration tests
├── configs/                # YAML configuration files
├── scripts/                # runnable end-to-end pipeline scripts
├── pyproject.toml
└── README.md
```

## Pipeline Flow

The end-to-end pipeline (`scripts/example_end_to_end_pipeline.py`) executes the following stages:

```
Reader → DataProcessor → ForecastingPipeline → Evaluator → MetricsSummary
  → ModelSelector → ForecastExtractor → Optimizer → Flow decisions
```

1. **Data Ingestion** — reads parquet files (demand history, origins, destinations, lanes)
2. **Data Processing** — validates, deduplicates, and sorts each dataset
3. **Forecasting** — runs Naive, Seasonal, and Rolling Window models
4. **Evaluation** — computes MAE, MSE, MAPE, WAPE per model
5. **Model Selection** — picks the best model by WAPE
6. **Demand Aggregation** — extracts forecasts and computes average daily demand per destination
7. **Optimization** — solves a min-cost transportation LP to allocate supply to destinations
8. **Output** — prints optimal flow allocation and total shipping cost

## Planned Features
 - Synthetic logistics network generator
 - Demand generation pipeline
 - Baseline forecasting model
 - Event-driven simulator
 - Network optimization model
 - API endpoints for end-to-end execution
 - MLflow experiment tracking
 - Performance benchmarking with Polars vs Pandas
 - Docker support

## Why This Project

This project is a portfolio piece built to demonstrate the ability to design and implement decision systems that go beyond isolated models.

It emphasizes:

- scalable data handling,
- integration between ML and optimization,
- software engineering discipline,
- reproducibility and deployability.

## Status

Core pipeline implemented and functional: data ingestion → forecasting → model selection → optimization.

## Next Steps
- Implement stochastic simulation layer
- Add FastAPI endpoints for end-to-end execution
- Integrate MLflow experiment tracking
- Add Docker support
- (Optional) integrate an AI powered layer to answer business related questions on data

Author

Christian Piermarini
Applied Scientist / Operations Research / Machine Learning
