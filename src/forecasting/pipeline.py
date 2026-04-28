"""
This class is responsible to get a list of models and then produce the output.
"""

import logging

import polars as pl
from forecasting.models.base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)


class ForecastingPipeline:

    def __init__(self, models: list[BaseForecaster]):
        self.models = models

    def run(self, df: pl.DataFrame, train_ratio: float | None = None) -> pl.DataFrame:
        result_df = df

        if train_ratio is not None:
            if train_ratio <= 0 or train_ratio >= 1:
                raise ValueError("train_ratio must be between 0 and 1 exclusive")
            split_idx = int(len(df) * train_ratio)
            train_df = df[:split_idx]
            logger.info(
                "Train/test split at index %d (train_ratio=%.4f, total rows=%d)",
                split_idx,
                train_ratio,
                len(df),
            )

            for model in self.models:
                model.fit(train_df)
                result_df = model.predict(result_df)
        else:
            for model in self.models:
                model.fit(result_df)
                result_df = model.predict(result_df)

        return result_df
