"""
Main class adopted to read YAML configuration file.

The structure fo the config would be the following:

[1] Main field:
    [2] Secondary field
[1] Second main field

etc.

Hence, the idea would be to create a single dataclass for each main field, with an attribute
for each secondary field. Then merge all together using a main Config dataclass.

"""

from dataclasses import dataclass
from pathlib import Path
import yaml

### ----- constants

KNOWN_METRICS = {"mae", "mse", "rmse", "mape", "wape"}

### ----- config related dataclasses


@dataclass
class DataConfig:
    input_path: Path


@dataclass
class ForecastingConfig:
    metric: str = "wape"
    train_ratio: float = 0.8


@dataclass
class Config:
    data: DataConfig
    forecasting: ForecastingConfig | None = None


### ----- main config reading method


def load_config(project_root: Path, config_path: Path) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)  ## raw is a dict

    forecasting_config: ForecastingConfig | None = None
    forecasting_raw = raw.get("forecasting")

    if forecasting_raw is not None:
        metric = forecasting_raw.get("metric", "wape")
        if metric not in KNOWN_METRICS:
            raise ValueError(
                f"Unrecognised metric '{metric}'. Must be one of {sorted(KNOWN_METRICS)}."
            )

        train_ratio = forecasting_raw.get("train_ratio", 0.8)
        forecasting_config = ForecastingConfig(
            metric=metric,
            train_ratio=train_ratio,
        )

    return Config(
        data=DataConfig(project_root / (raw["data"]["input_path"])),
        forecasting=forecasting_config,
    )
