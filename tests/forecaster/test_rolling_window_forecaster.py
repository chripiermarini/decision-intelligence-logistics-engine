import pytest
import polars as pl
from forecasting.rolling_window_forecaster import RollingWindowForecaster


class TestRollingWindowForecaster:

    def test_rolling_window(self):
        df = pl.DataFrame(
            {
                "demand": [
                    48.87,
                    43.73,
                    47.48,
                    41.6,
                    46.25,
                    44.57,
                    40.87,
                    58.83,
                ]
            }
        )

        forecaster = RollingWindowForecaster(rolling_window=7, target_col="demand")

        result = forecaster.predict(df)

        forecast = result["ma_7_forecast"].to_list()

        assert forecast[:7] == [None] * 7

        expected_mean = (
            sum(
                [
                    48.87,
                    43.73,
                    47.48,
                    41.6,
                    46.25,
                    44.57,
                    40.87,
                ]
            )
            / 7
        )

        assert forecast[7] == pytest.approx(expected_mean, rel=1e-6)
