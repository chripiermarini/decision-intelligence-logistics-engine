from polars import DataFrame


class BaseProcessor:

    @staticmethod
    def _validate_non_empty(df: DataFrame) -> None:
        if df.is_empty():
            raise ValueError("Empty dataset")

    @staticmethod
    def _validate_no_nulls(df: DataFrame) -> None:
        if df.null_count().to_numpy().sum() > 0:
            raise ValueError("Null values are not allowed.")

    @staticmethod
    def _validate_columns(df: DataFrame, required_cols: set[str]) -> None:
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols}")
