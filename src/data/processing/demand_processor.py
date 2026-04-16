from data.processing.base_processor import BaseProcessor

from polars import DataFrame


class DemandProcessor(BaseProcessor):

    REQUIRED_COLUMNS = {"date", "destination_id", "demand"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        DemandProcessor._validate_non_empty(df)
        DemandProcessor._validate_no_nulls(df)
        DemandProcessor._validate_columns(df, DemandProcessor.REQUIRED_COLUMNS)

        df = df.unique()
        df = df.sort(["destination_id", "date"])

        return df
