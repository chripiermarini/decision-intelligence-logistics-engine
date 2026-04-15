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

### ----- config related dataclasses


@dataclass
class DataConfig:
    input_path: Path


@dataclass
class Config:
    data: DataConfig


### ----- main config reading method


def load_config(project_root: Path, config_path: Path) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)  ## raw is a dict

    return Config(
        DataConfig(project_root / (raw["data"]["input_path"])),
    )
