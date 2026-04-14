from __future__ import annotations

from pathlib import Path
import itertools

from datetime import datetime
import numpy as np
import polars as pl


def generate_synthetic_logistics_data(
    output_dir: Path,
    n_origins: int,
    n_destinations: int,
    start_date: datetime,
    end_date: datetime,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    origin_ids = [f"O{i:02d}" for i in range(1, n_origins + 1)]
    destination_ids = [f"D{i:02d}" for i in range(1, n_destinations + 1)]

    origins = pl.DataFrame(
        {
            "origin_id": origin_ids,
            "region": rng.choice(["North", "South", "Central"], size=n_origins),
            "daily_capacity": rng.integers(80, 180, size=n_origins),
        }
    )

    destinations = pl.DataFrame(
        {
            "destination_id": destination_ids,
            "region": rng.choice(["North", "South", "Central"], size=n_destinations),
            "base_demand_level": rng.integers(20, 70, size=n_destinations),
        }
    )

    lanes_rows = []
    for origin_id, destination_id in itertools.product(origin_ids, destination_ids):
        lanes_rows.append(
            {
                "origin_id": origin_id,
                "destination_id": destination_id,
                "unit_cost": float(rng.integers(5, 25)),
                "lead_time_days": int(rng.integers(1, 5)),
                "max_lane_capacity": int(rng.integers(30, 120)),
            }
        )

    lanes = pl.DataFrame(lanes_rows)

    dates = pl.date_range(start=start_date, end=end_date, interval="1d", eager=True)

    demand_rows = []
    base_lookup = {
        row["destination_id"]: row["base_demand_level"]
        for row in destinations.to_dicts()
    }

    for destination_id in destination_ids:
        base = base_lookup[destination_id]
        for i, date in enumerate(dates):
            dow = date.weekday()
            is_weekend = dow >= 5

            weekly_seasonality = 0.85 if is_weekend else 1.0
            trend = 1.0 + 0.0015 * i
            promo_flag = int(rng.random() < 0.08)
            promo_multiplier = 1.25 if promo_flag else 1.0
            noise = rng.normal(0, 4)

            demand = max(
                0,
                base * weekly_seasonality * trend * promo_multiplier + noise,
            )

            demand_rows.append(
                {
                    "date": date,
                    "destination_id": destination_id,
                    "demand": round(float(demand), 2),
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "promo_flag": promo_flag,
                }
            )

    demand_history = pl.DataFrame(demand_rows)

    origins.write_parquet(output_dir / "origins.parquet")
    destinations.write_parquet(output_dir / "destinations.parquet")
    lanes.write_parquet(output_dir / "lanes.parquet")
    demand_history.write_parquet(output_dir / "demand_history.parquet")
