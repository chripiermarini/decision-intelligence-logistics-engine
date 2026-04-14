from pathlib import Path

from data.generation import generate_synthetic_logistics_data
from datetime import datetime

def main() -> None:
    output_dir = Path("../data/synthetic")
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


if __name__ == "__main__":
    main()