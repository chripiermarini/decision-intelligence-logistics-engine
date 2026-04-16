"""
In this module, we are going to implement a simple moving average class module.
The module will allow to use a naive baseline model computed through a rolling average.
"""

import polars as pl
from forecasting.base_forecaster import BaseForecaster

class RollingWindowForecaster(BaseForecaster):

    def __init__(self, rolling_window : int):

        self.rolling_window = rolling_window

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

        forecast_col = f"ma_{self.rolling_window}_forecast"

        return (
            df.with_columns(
                pl.col("demand")
                .shift(1)
                .rolling_mean(window_size=self.rolling_window)
                .alias(forecast_col)
            )
        )
