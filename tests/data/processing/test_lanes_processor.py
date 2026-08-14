import polars as pl
import pytest

from data.processing.lanes_processor import LanesProcessor


class TestLanesProcessor:
    @pytest.fixture
    def valid_lanes_df(self):
        return pl.DataFrame(
            {
                "origin_id": ["O1", "O2"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [10.0, 15.0],
            }
        )

    def test_process_valid_lanes(self, valid_lanes_df):
        df = LanesProcessor.process(valid_lanes_df)
        assert df is not None
        assert df.height == 2

    def test_process_with_lead_time_days(self, valid_lanes_df):
        df_with_lt = valid_lanes_df.with_columns(pl.Series("lead_time_days", [2, 0]))
        df = LanesProcessor.process(df_with_lt)
        assert "lead_time_days" in df.columns
        assert set(df["lead_time_days"].to_list()) == {2, 0}

    def test_process_negative_lead_time_days_raises(self, valid_lanes_df):
        invalid_df = valid_lanes_df.with_columns(pl.Series("lead_time_days", [-1, 2]))
        with pytest.raises(ValueError, match="Negative lead_time_days values found"):
            LanesProcessor.process(invalid_df)

    def test_process_empty_df_raises(self):
        empty_df = pl.DataFrame({})
        with pytest.raises(ValueError):
            LanesProcessor.process(empty_df)
