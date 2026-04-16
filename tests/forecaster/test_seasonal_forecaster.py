import pytest
import polars as pl

from forecasting.seasonal_forecaster import SeasonalForecaster


class TestSeasonalForecaster:
    def test_seasonal_forecaster(self):
        valid_df = pl.DataFrame(
            {"demand": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        )

        lag_value = 7
        seasonal_forecaster = SeasonalForecaster(lag_value=lag_value)
        forecasted_df = seasonal_forecaster.predict(valid_df)

        assert forecasted_df.shape == (15, 2)

        assert forecasted_df[7, f"seasonal_forecast_lag_{lag_value}"] == 1
