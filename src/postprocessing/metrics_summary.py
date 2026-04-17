from pathlib import Path
import polars as pl


class MetricsSummary:

    def __init__(self, output_folder_path: Path | None = None):
        self.output_folder_path = output_folder_path
        self.models_results: list[dict] = []

    def collect(self, model_name: str, results: dict):
        """
        Add results for a given model.
        """
        row = {"model_name": model_name} | results
        self.models_results.append(row)

    def produce_summary(
        self, sort_by: str | None = "wape", ascending: bool = True
    ) -> pl.DataFrame:
        """
        Build a normalized summary table.

        Parameters:
        - sort_by: column name to sort by (e.g. 'wape')
        - ascending: sort order
        """
        if not self.models_results:
            return pl.DataFrame()

        self._normalize_schema()
        df = pl.DataFrame(self.models_results)

        # Optional sorting
        if sort_by and sort_by in df.columns:
            df = df.sort(sort_by, descending=not ascending)

        return df

    def save_summary(self, df: pl.DataFrame):
        df.write_csv(self.output_folder_path / "metrics_summary.csv")

    def _normalize_schema(self):
        """
        Ensure all rows have the same keys (schema alignment).
        Missing keys are filled with None.
        """

        all_keys = set()
        for row in self.models_results:
            all_keys.update(row.keys())

        for row in self.models_results:
            for key in all_keys:
                if key not in row:
                    row[key] = None
