"""Forecast extraction and demand aggregation utilities."""

import polars as pl


class ForecastExtractor:
    """Stateless utilities for extracting forecasts and aggregating demand."""

    @staticmethod
    def extract(results_df: pl.DataFrame, forecast_col: str) -> pl.DataFrame:
        """Return a DataFrame with columns ``[date, destination_id, forecast]``.

        Parameters
        ----------
        results_df : pl.DataFrame
            Full forecasting results containing at least ``date``,
            ``destination_id``, and the column named *forecast_col*.
        forecast_col : str
            Name of the column holding the selected model's predictions.

        Returns
        -------
        pl.DataFrame
            A three-column DataFrame with the forecast column renamed to
            ``forecast``.  All rows are preserved, including those with
            null forecast values.

        Raises
        ------
        ValueError
            If any of the required columns are missing from *results_df*.
        """
        required = {"date", "destination_id", forecast_col}
        missing = sorted(required - set(results_df.columns))
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {results_df.columns}"
            )

        return results_df.select(
            pl.col("date"),
            pl.col("destination_id"),
            pl.col(forecast_col).alias("forecast"),
        )

    @staticmethod
    def aggregate_demand(forecast_df: pl.DataFrame) -> pl.DataFrame:
        """Return a DataFrame with columns ``[destination_id, demand]``.

        Groups *forecast_df* by ``destination_id`` and sums the
        ``forecast`` column, excluding null values.  The aggregated
        column is renamed to ``demand``.

        Parameters
        ----------
        forecast_df : pl.DataFrame
            A Forecast_DataFrame with schema
            ``[date, destination_id, forecast]``.

        Returns
        -------
        pl.DataFrame
            One row per destination with the total demand.
        """
        return (
            forecast_df.group_by("destination_id")
            .agg(pl.col("forecast").sum().alias("demand"))
        )
