"""
In this module, we are going to implement a simple moving average class module.
The module will allow to use a naive baseline model computed through a rolling average.
"""

import polars as pl
from forecasting.models.base_forecaster import BaseForecaster


class RollingWindowForecaster(BaseForecaster):

    def __init__(self, target_col: str = "demand", rolling_window: int = 1):

        self.rolling_window = rolling_window
        self.target_col = target_col
        self.forecast_col = f"ma_{self.rolling_window}_forecast"

    @property
    def name(self):
        return f"ma_{self.rolling_window}_forecaster"

    def fit(self, df: pl.DataFrame):
        """
        Rolling window forecasting do not require training.
        """
        pass

    def predict(self, df: pl.DataFrame):
        """
        Main rolling window method, taking as input the dataframe we want to predict on, we compute the
        moving average prediction using a rolling average.
        """

        return df.with_columns(
            pl.col(self.target_col)
            .shift(1)
            .rolling_mean(window_size=self.rolling_window)
            .alias(self.forecast_col)
        )
