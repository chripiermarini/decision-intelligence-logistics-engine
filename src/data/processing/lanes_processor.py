from data.processing.base_processor import BaseProcessor

from polars import DataFrame


class LanesProcessor(BaseProcessor):

    REQUIRED_COLUMNS = {"origin_id", "destination_id", "unit_cost"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        LanesProcessor._validate_non_empty(df)
        LanesProcessor._validate_columns(df, LanesProcessor.REQUIRED_COLUMNS)

        df = df.unique()

        return df
