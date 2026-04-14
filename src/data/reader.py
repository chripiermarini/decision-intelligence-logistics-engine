from data.input_data import InputData
from pathlib import Path
import polars as pl
from polars import DataFrame


class Reader:

    def __init__(self, input_file_path: str | Path):
        self.input_file_path = Path(input_file_path)

        if not self.input_file_path.exists(): ## ! CAREFUL !
            raise ValueError(f"Path does not exist: {self.input_file_path}")

    def read_input_files(self) -> InputData:
        return InputData(
            demand_history=self.read_demand_history_file(),
            destinations=self.read_destinations_file(),
            lanes=self.read_lanes_file(),
            origins=self.read_origins_file(),
        )

    def read_demand_history_file(self) -> DataFrame:
        return self._read_parquet("demand_history.parquet")

    def read_destinations_file(self) -> DataFrame:
        return self._read_parquet("destinations.parquet")

    def read_lanes_file(self) -> DataFrame:
        return self._read_parquet("lanes.parquet")

    def read_origins_file(self) -> DataFrame:
        return self._read_parquet("origins.parquet")

    def _read_parquet(self, filename: str) -> DataFrame:
        file_path = self.input_file_path / filename

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        return pl.read_parquet(file_path)