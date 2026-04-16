"""
Main evaluator class, computing main forecasting accuracy metrics given a dataframe and the names of the
target column and the forecast column.
"""

import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


class Evaluator:

    def __init__(self, df: pl.DataFrame, target_col_name: str, forecast_col_name: str):
        self.target_col_name = target_col_name
        self.forecast_col_name = forecast_col_name
        self.df = self._delete_null_values(df)

    def compute_metrics(self):

        target = self.df[self.target_col_name].to_numpy()
        forecast = self.df[self.forecast_col_name].to_numpy()

        mae = mean_absolute_error(target, forecast)
        mse = mean_squared_error(target, forecast)
        mape = mean_absolute_percentage_error(target, forecast)
        wape = self._wape(target, forecast)

        return {
            "mae": mae,
            "mse": mse,
            "mape": mape,
            "wape": wape,
        }

    @staticmethod
    def _wape(target_col, forecast_col):
        denominator = np.sum(np.abs(target_col))
        if denominator == 0:
            return np.nan

        return np.sum(np.abs(target_col - forecast_col)) / denominator

    def _delete_null_values(self, df: pl.DataFrame):
        original_length = df.height

        filtered_df = df.drop_nulls(subset=[self.forecast_col_name])
        if original_length != filtered_df.height:
            print("Null values found and removed.")

        return filtered_df
