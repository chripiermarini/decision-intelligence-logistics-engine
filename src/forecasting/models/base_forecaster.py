"""
Base forecaster model made to be considered as an abstract interface for the
other main forecaster module.
"""

import polars as pl
from abc import ABC, abstractmethod


class BaseForecaster(ABC):

    @abstractmethod
    def fit(self, df: pl.DataFrame):
        """
        Main class designed for machine learning models that require training.
        """
        pass

    @abstractmethod
    def predict(self, df: pl.DataFrame):
        """
        Main designed to perform prediction given the input
        """
        pass
