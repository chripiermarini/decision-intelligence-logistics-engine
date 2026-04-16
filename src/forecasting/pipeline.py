"""
This class is responsible to get a list of models and then produce the output.
"""

import polars as pl
from forecasting.models.base_forecaster import BaseForecaster


class ForecastingPipeline:

    def __init__(self, models: list[BaseForecaster]):
        self.models = models

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        result_df = df

        for model in self.models:
            model.fit(result_df)
            result_df = model.predict(result_df)

        return result_df
