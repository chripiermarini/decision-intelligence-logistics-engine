"""
Example end-to-end pipeline for the Decision Intelligence Logistics Engine.

This script demonstrates the full workflow:
- data ingestion
- data processing
- forecasting
- model evaluation
- model selection
- demand aggregation
- optimization
- visualization

Usage:
    python example_end_to_end_pipeline.py --config configs/test_config.yaml
"""

import argparse
import logging

from data.ingestion import Reader
from data.processing.data_processor import DataProcessor

from forecasting.models.naive_forecaster import NaiveForecaster
from forecasting.models.rolling_window_forecaster import RollingWindowForecaster
from forecasting.models.seasonal_forecaster import SeasonalForecaster
from forecasting.pipeline import ForecastingPipeline
from forecasting.evaluator import Evaluator
from forecasting.forecast_extractor import ForecastExtractor
from forecasting.model_selector import ModelSelector

from postprocessing.metrics_summary import MetricsSummary
from postprocessing.visualization import VisualizationEngine

from optimization import Optimizer

from utils.config import load_config
from utils.system_paths import get_project_root

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_config.yaml",
        help="Path to config file",
    )
    return parser.parse_args()


def run_forecasting(clean_data, models):
    pipeline = ForecastingPipeline(models=models)
    results = pipeline.run(clean_data.demand_history)

    logger.info("Forecasting completed. Shape: %s", results.shape)
    return results


def evaluate_models(results, models, output_path):
    evaluator = Evaluator(results, "demand")
    metrics_summary = MetricsSummary(output_folder_path=output_path)

    for model in models:
        metrics_results = evaluator.compute_metrics(
            forecast_col_name=model.forecast_col
        )
        metrics_summary.collect(model_name=model.name, results=metrics_results)

    summary = metrics_summary.produce_summary()
    metrics_summary.save_summary(summary)

    logger.info("Model evaluation completed.")
    return summary


def select_and_aggregate(results, summary, models):
    best_model_name = ModelSelector.select_best(summary, metric="wape")
    best_model = next(m for m in models if m.name == best_model_name)

    logger.info("Best model selected: %s", best_model_name)

    forecast_df = ForecastExtractor.extract(results, best_model.forecast_col)
    demand_df = ForecastExtractor.aggregate_average_demand(forecast_df)

    return best_model, demand_df


# ---------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------
def run_optimization(demand_df, origins_df, lanes_df):
    optimizer = Optimizer(solver_name="GLOP")

    result = optimizer.solve(
        demand_df=demand_df,
        origins_df=origins_df,
        lanes_df=lanes_df,
    )

    logger.info("Optimization completed. Total cost: %.4f", result.total_cost)
    return result


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def run_visualization(results, best_model, project_root):
    visualization = VisualizationEngine(df=results)

    visualization.produce_timeseries_plots(
        target_destination="D01",
        actuals_col_name="demand",
        predicted_col_name=best_model.forecast_col,
        save_fig_location=project_root / "data" / "output" / "plots",
    )

    logger.info("Visualization generated.")


def main():
    args = parse_args()

    project_root = get_project_root()
    config = load_config(project_root, project_root / args.config)

    # --- Data ingestion & processing ---
    reader = Reader(config.data.input_path)
    raw_data = reader.read()
    clean_data = DataProcessor.process(raw_data)

    if clean_data.demand_history.is_empty():
        raise ValueError("Empty demand history after processing")

    # --- Forecasting ---
    models = [
        NaiveForecaster(),
        SeasonalForecaster(lag_value=7),
        RollingWindowForecaster(rolling_window=7),
    ]

    results = run_forecasting(clean_data, models)

    # --- Evaluation ---
    summary = evaluate_models(
        results,
        models,
        project_root / "data" / "output",
    )

    # --- Model selection + demand ---
    best_model, demand_df = select_and_aggregate(results, summary, models)

    # --- Optimization ---
    opt_result = run_optimization(
        demand_df=demand_df,
        origins_df=clean_data.origins,
        lanes_df=clean_data.lanes,
    )

    logger.info("Flows:\n%s", opt_result.flows)

    # --- Visualization ---
    run_visualization(results, best_model, project_root)

    return opt_result


if __name__ == "__main__":
    main()
