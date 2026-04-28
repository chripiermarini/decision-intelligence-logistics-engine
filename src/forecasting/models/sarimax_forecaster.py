"""
ARIMA/SARIMA forecaster wrapping statsmodels SARIMAX.
Supports configurable order (p,d,q) and optional seasonal_order (P,D,Q,s).
"""

import numpy as np
import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.models.base_forecaster import BaseForecaster


class ARIMAForecaster(BaseForecaster):

    def __init__(
        self,
        target_col: str = "demand",
        order: tuple[int, int, int] = (1, 1, 0),
        seasonal_order: tuple[int, int, int, int] | None = None,
    ):
        self.target_col = target_col
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_col = "arima_forecast"
        self._fitted_result = None
        self._train_length = 0

    @property
    def name(self):
        if self.seasonal_order is not None:
            return "sarima_forecaster"
        return "arima_forecaster"

    def fit(self, df: pl.DataFrame):
        if self.target_col not in df.columns:
            raise ValueError(f"{self.target_col} not in dataframe")

        y = df[self.target_col].to_numpy().astype(float)
        self._train_length = len(y)

        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=(
                self.seasonal_order if self.seasonal_order is not None else (0, 0, 0, 0)
            ),
        )
        self._fitted_result = model.fit(disp=False)

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.target_col not in df.columns:
            raise ValueError(f"{self.target_col} not in dataframe")

        n = len(df)
        fitted_vals = self._fitted_result.fittedvalues

        if n <= self._train_length:
            forecast_values = fitted_vals[:n]
        else:
            oos_steps = n - self._train_length
            oos_forecast = self._fitted_result.forecast(steps=oos_steps)
            forecast_values = np.concatenate([fitted_vals, oos_forecast])

        return df.with_columns(
            pl.Series(name=self.forecast_col, values=forecast_values)
        )
