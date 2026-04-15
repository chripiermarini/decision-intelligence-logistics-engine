import polars as pl
import pytest
from data.ingestion import DemandProcessor


class TestDemandProcessor:

    @pytest.fixture
    def valid_df(self):
        return pl.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "destination_id": ["D1", "D1"],
                "demand": [10.0, 20.0],
            }
        )

    def test_valid_df(self, valid_df):
        processor = DemandProcessor.process(valid_df)
        assert processor.height == 2

    def test_empty_df(self):
        empty_df = pl.DataFrame({})
        with pytest.raises(ValueError):
            DemandProcessor.process(empty_df)

    def test_null_values(self):

        null_values = pl.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "destination_id": ["D1", "D1"],
                "demand": [None, None],
            }
        )

        with pytest.raises(ValueError):
            DemandProcessor.process(null_values)
