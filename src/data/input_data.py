"""
Input data datclass, allowing to gather the Polars dataframe in one single practical dataclass
"""

from dataclasses import dataclass
import polars as pl


@dataclass
class InputData:
    demand_history: pl.DataFrame
    destinations: pl.DataFrame
    lanes: pl.DataFrame
    origins: pl.DataFrame
