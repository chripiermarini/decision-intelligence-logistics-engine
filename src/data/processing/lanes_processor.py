"""Validation and cleaning for the lanes reference table."""

import polars as pl

from data.processing.validation import validate_columns, validate_non_empty


class LanesProcessor:
    """Validates and deduplicates the lanes reference table.

    Required columns: ``origin_id``, ``destination_id``, ``unit_cost``.
    """

    REQUIRED_COLUMNS: set[str] = {"origin_id", "destination_id", "unit_cost"}

    @staticmethod
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Validate and deduplicate the lanes table.

        Parameters
        ----------
        df : pl.DataFrame
            Raw lanes DataFrame.

        Returns
        -------
        pl.DataFrame
            Cleaned lanes table with duplicate rows removed.

        Raises
        ------
        ValueError
            If the DataFrame is empty or missing required columns, or if lead_time_days is negative.
        """
        validate_non_empty(df)
        validate_columns(df, LanesProcessor.REQUIRED_COLUMNS)

        if "lead_time_days" in df.columns:
            negative = df.filter(pl.col("lead_time_days") < 0)
            if not negative.is_empty():
                invalid = negative.select(
                    "origin_id", "destination_id", "lead_time_days"
                ).to_dicts()
                raise ValueError(f"Negative lead_time_days values found: {invalid}")

        return df.unique()
