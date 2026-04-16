from data.processing.base_processor import BaseProcessor
from polars import DataFrame


class OriginsProcessor(BaseProcessor):

    REQUIRED_COLUMNS = {"origin_id"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        OriginsProcessor._validate_non_empty(df)
        OriginsProcessor._validate_columns(df, OriginsProcessor.REQUIRED_COLUMNS)

        return df.unique()
