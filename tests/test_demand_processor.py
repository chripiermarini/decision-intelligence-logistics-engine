import pytest
import polars as pl

from data.processing.demand_processor import DemandProcessor

class TestDemandProcessor:

    @pytest.fixture
    def valid_frame(self):
        return pl.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "destination_id": ["D1", "D1"],
                "demand": [10.0, 20.0],
            }
        )

    def test_validate(self, valid_frame):
        df = DemandProcessor.process(valid_frame)

        assert df is not None
        assert df.shape[0] == 2


