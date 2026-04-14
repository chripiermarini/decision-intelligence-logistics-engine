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
- Demand prediction using baseline ML models
- Feature engineering with lag and rolling statistics
- Experiment tracking with MLflow

### 3. Simulation Layer
- Event-driven simulation of shipment arrivals, delays, and processing
- Stochastic demand generation
- Scenario analysis under uncertainty

### 4. Optimization Layer
- Network flow / origin-destination assignment model
- Capacity-constrained cost minimization
- Integration of forecast outputs into downstream decision-making

### 5. Serving Layer
- FastAPI endpoints for simulation, forecasting, and optimization
- Reproducible configuration and modular architecture

## Initial Tech Stack

- Python
- Polars
- DuckDB
- Pyomo
- FastAPI
- MLflow
- Pandas / NumPy
- SimPy (optional, for simulation extensions)

## Repository Structure

```text
decision-intelligence-logistics-engine/
│
├── data/               # raw, interim, and processed datasets
├── notebooks/          # exploratory analysis and prototyping
├── src/
│   ├── data/           # ingestion, preprocessing, validation
│   ├── forecasting/    # feature engineering and ML models
│   ├── simulation/     # stochastic and event-driven simulation
│   ├── optimization/   # mathematical models and solvers
│   ├── api/            # FastAPI app and endpoints
│   └── utils/          # shared utilities
│
├── tests/              # unit and integration tests
├── configs/            # yaml/json configuration files
├── scripts/            # runnable scripts for training and pipelines
├── requirements.txt
└── README.md
```

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

Project setup in progress.

## Next Steps
- Define the synthetic logistics network schema
- Create the initial data generation pipeline
- Implement a baseline optimization model
- Add the first FastAPI endpoint

Author

Christian Piermarini
Applied Scientist / Operations Research / Machine Learning
