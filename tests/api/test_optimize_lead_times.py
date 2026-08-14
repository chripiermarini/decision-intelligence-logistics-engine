"""API tests for optional lane lead_time_days support on /optimize and /plan."""

from fastapi.testclient import TestClient
import pytest

from api.app import app, _to_lanes_df
from api.models import LaneRecord

client = TestClient(app)


class TestToLanesDf:
    def test_omits_lead_time_days_column_when_all_none(self):
        df = _to_lanes_df(
            [
                LaneRecord(origin_id="O1", destination_id="D1", unit_cost=1.0),
                LaneRecord(origin_id="O1", destination_id="D2", unit_cost=2.0),
            ]
        )
        assert "lead_time_days" not in df.columns or (df["lead_time_days"] == 0).all()

    def test_includes_lead_time_days_default_zero(self):
        df = _to_lanes_df(
            [
                LaneRecord(
                    origin_id="O1",
                    destination_id="D1",
                    unit_cost=1.0,
                    lead_time_days=2,
                ),
                LaneRecord(origin_id="O1", destination_id="D2", unit_cost=2.0),
            ]
        )
        assert "lead_time_days" in df.columns
        by_dest = {
            row["destination_id"]: row["lead_time_days"] for row in df.to_dicts()
        }
        assert by_dest == {"D1": 2, "D2": 0}


class TestOptimizeLeadTimes:
    def test_optimize_with_1day_lead_time(self):
        # Demand on 2026-06-01: 0, 2026-06-02: 10
        # Lane lead time = 1 day
        # To satisfy 2026-06-02 demand, shipment must be dispatched on 2026-06-01
        payload = {
            "demand_ts": [
                {"date": "2026-06-01", "destination_id": "D1", "demand": 0.0},
                {"date": "2026-06-02", "destination_id": "D1", "demand": 10.0},
            ],
            "origins": [{"origin_id": "O1", "daily_capacity": 100.0}],
            "lanes": [
                {
                    "origin_id": "O1",
                    "destination_id": "D1",
                    "unit_cost": 5.0,
                    "lead_time_days": 1,
                }
            ],
            "destinations": [{"destination_id": "D1", "holding_cost": 0.0}],
            "initial_inventory": {"D1": 0.0},
        }
        response = client.post("/optimize", json=payload)
        assert response.status_code == 200
        body = response.json()
        flows = body["flows"]
        # Flow dispatched on 2026-06-01 should be 10.0
        f_day1 = next(
            f
            for f in flows
            if f["period"] == "2026-06-01"
            and f["origin_id"] == "O1"
            and f["destination_id"] == "D1"
        )
        assert f_day1["flow"] == 10.0

    def test_negative_lead_time_raises_422(self):
        payload = {
            "demand_ts": [
                {"date": "2026-06-01", "destination_id": "D1", "demand": 10.0}
            ],
            "origins": [{"origin_id": "O1", "daily_capacity": 100.0}],
            "lanes": [
                {
                    "origin_id": "O1",
                    "destination_id": "D1",
                    "unit_cost": 1.0,
                    "lead_time_days": -1,
                }
            ],
            "destinations": [{"destination_id": "D1"}],
        }
        response = client.post("/optimize", json=payload)
        assert response.status_code == 422
