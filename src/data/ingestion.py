from pathlib import Path
import polars as pl
from polars import DataFrame

from data.input_data import InputData


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
