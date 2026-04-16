import polars

from forecasting.models.naive_forecaster import NaiveForecaster


class TestNaiveForecaster:

    def test_valid_df_naive_forecaster(self):

        valid_df = polars.DataFrame({"demand": [10.2, 13, 14.5]})

        forecaster = NaiveForecaster(target_col="demand")

        forecast_df = forecaster.predict(valid_df)

        assert forecast_df.shape == (3, 2)  ## rows first, columns second

        assert forecast_df[1, "naive_forecast"] == 10.2
