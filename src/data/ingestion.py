from pathlib import Path
import polars as pl
from polars import DataFrame

from data.input_data import InputData


class DemandProcessor:
    """
    Scope of this class is to reduce the redundancy of the input data, provide the correct format and type
    of the data, and handle possible null values.
    The idea is to implement all the required modification of the data in order to allow the model
    to ingest them and use them.
    """

    @staticmethod
    def process(df: DataFrame) -> DataFrame:

        DemandProcessor._validate_non_empty_dataset(df)
        DemandProcessor._validate_no_nulls(df)
        DemandProcessor._validate_columns(df)
        df = DemandProcessor._remove_duplicates(df)

        return df

    @staticmethod
    def _validate_no_nulls(df: DataFrame) -> None:
        if df.null_count().to_numpy().sum() > 0:
            raise ValueError("Null values are not allowed.")

    @staticmethod
    def _validate_non_empty_dataset(df: DataFrame) -> None:
        if df.is_empty():
            raise ValueError("Empty dataset")

    @staticmethod
    def _validate_columns(df: DataFrame) -> None:
        required_cols = {"date", "destination_id", "demand"}

        if not required_cols.issubset(df.columns):
            raise ValueError("Missing required columns")

    @staticmethod
    def _remove_duplicates(df: DataFrame) -> DataFrame:
        original_rows = df.height
        unique_df = df.unique()

        if unique_df.height < original_rows:
            print("Duplicate entries found and removed.")

        return unique_df


class Reader:
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self._validate_path()

    def read(self) -> InputData:

        return InputData(
            demand_history=self._read_parquet("demand_history.parquet"),
            destinations=self._read_parquet("destinations.parquet"),
            lanes=self._read_parquet("lanes.parquet"),
            origins=self._read_parquet("origins.parquet"),
        )

    def _validate_path(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

    def _read_parquet(self, filename: str) -> DataFrame:
        file_path = self.input_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        return pl.read_parquet(file_path)
