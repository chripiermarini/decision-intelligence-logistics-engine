import polars as pl
from forecasting.models.base_forecaster import BaseForecaster


class NaiveForecaster(BaseForecaster):

    def __init__(self, target_col: str = "demand"):
        self.target_col = target_col
        self.forecast_col = "naive_forecast"

    def fit(self, df: pl.DataFrame):
        pass

    def predict(self, df: pl.DataFrame):

        if self.target_col not in df.columns:
            raise ValueError(f"{self.target_col} not in dataframe")

        return df.with_columns(
            [pl.col(self.target_col).shift(1).alias(self.forecast_col)]
        )
