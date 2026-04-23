from pathlib import Path

from data.generation import (
    generate_synthetic_logistics_data,
)
from datetime import datetime


def first_synthetic_dataset() -> None:
    output_dir = Path("../data/synthetic1")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_synthetic_logistics_data(
        output_dir=output_dir,
        n_origins=4,
        n_destinations=8,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30),
        seed=42,
    )

    print(f"Synthetic data generated in: {output_dir.resolve()}")


def second_synthetic_dataset() -> None:
    output_dir = Path("../data/synthetic2")
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_synthetic_logistics_data(
        output_dir=output_dir,
        n_origins=3,
        n_destinations=6,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30),
        seed=42,
    )


if __name__ == "__main__":
    second_synthetic_dataset()
