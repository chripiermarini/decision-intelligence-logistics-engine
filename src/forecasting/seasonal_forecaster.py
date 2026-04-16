"""
This class implements a naive seasonal forecaster, using the same value
of demand of the 'x'-days before.
"""

import polars as pl

from forecasting.base_forecaster import BaseForecaster


class SeasonalForecaster(BaseForecaster):

    def __init__(self, target_col: str = "demand", lag_value: int = 7):
        self.lag_value = lag_value
        self.target_col = target_col

        self.forecast_col = f"seasonal_forecast_lag_{lag_value}"

    def fit(self, df: pl.DataFrame):
        pass

    def predict(self, df: pl.DataFrame):
        return df.with_columns(
            [
                pl.col(self.target_col).shift(self.lag_value).alias(self.forecast_col),
            ]
        )
