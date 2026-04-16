from data.processing.base_processor import BaseProcessor

from polars import DataFrame


class DestinationsProcessor(BaseProcessor):

    REQUIRED_COLUMNS = {"destination_id"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        DestinationsProcessor._validate_non_empty(df)
        DestinationsProcessor._validate_columns(
            df, DestinationsProcessor.REQUIRED_COLUMNS
        )

        return df.unique()
