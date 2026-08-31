"""Concrete implementation of APIInterface for the logistics engine."""

from datetime import date

import polars as pl

from api.api_interface import APIInterface
from forecasting import AggregatedForecastingResult, create_forecasting_pipeline
from optimization import MultiPeriodOptimizer, MultiPeriodResult
from forecasting.config import PerDestinationConfig


class LogisticsAPI(APIInterface):
    """Concrete API implementation wiring PerDestinationForecastingPipeline and MultiPeriodOptimizer.

    Parameters
    ----------
    config : PerDestinationConfig | None
        Forecasting pipeline configuration (model names, train ratio, etc.).
        Required for ``forecast``; optional for ``optimize`` -only usage.
    solver_name : str
        OR-Tools backend solver (default ``"GLOP"``).
    """

    def __init__(
        self,
        config: PerDestinationConfig | None = None,
        solver_name: str = "GLOP",
    ) -> None:
        self._config = config
        self._solver_name = solver_name

    def forecast(self, input_data: pl.DataFrame) -> AggregatedForecastingResult:
        """Run per-destination forecasting on historical demand data."""
        config = self._config
        if config is None:
            raise ValueError("Cannot forecast without a PerDestinationConfig. Pass config= when constructing LogisticsAPI.")
        pipeline = create_forecasting_pipeline(config)
        return pipeline.run(input_data)

    def optimize(
        self,
        demand_ts: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame,
        planning_horizon: list[date],
        initial_inventory: dict[str, float] | None = None,
    ) -> MultiPeriodResult:
        """Solve the multi-period transportation LP."""
        optimizer = MultiPeriodOptimizer(solver_name=self._solver_name)
        return optimizer.solve(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
            initial_inventory=initial_inventory,
        )
