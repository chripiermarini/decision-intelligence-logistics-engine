import pytest
import numpy as np
import polars as pl

from forecasting.evaluator import Evaluator


class TestEvaluator:

    def test_compute_metrics_basic(self):
        df = pl.DataFrame({
            "demand": [10.0, 20.0, 30.0],
            "forecast": [12.0, 18.0, 33.0],
        })

        evaluator = Evaluator(df, "demand", "forecast")
        metrics = evaluator.compute_metrics()

        assert "mae" in metrics
        assert "mse" in metrics
        assert "mape" in metrics
        assert "wape" in metrics

        errors = np.array([2.0, 2.0, 3.0])

        assert metrics["mae"] == pytest.approx(np.mean(np.abs(errors)))
        assert metrics["mse"] == pytest.approx(np.mean(errors**2))
        assert metrics["wape"] == pytest.approx(np.sum(np.abs(errors)) / np.sum(df["demand"].to_numpy()))


    def test_null_values_are_removed(self):
        df = pl.DataFrame({
            "demand": [10.0, 20.0, 30.0],
            "forecast": [None, 18.0, 33.0],
        })

        evaluator = Evaluator(df, "demand", "forecast")
        metrics = evaluator.compute_metrics()

        expected_errors = np.array([2.0, 3.0])

        assert metrics["mae"] == pytest.approx(np.mean(np.abs(expected_errors)))


    def test_wape_zero_denominator(self):
        df = pl.DataFrame({
            "demand": [0.0, 0.0, 0.0],
            "forecast": [1.0, 2.0, 3.0],
        })

        evaluator = Evaluator(df, "demand", "forecast")
        metrics = evaluator.compute_metrics()

        assert np.isnan(metrics["wape"])


    def test_perfect_forecast(self):
        df = pl.DataFrame({
            "demand": [10.0, 20.0, 30.0],
            "forecast": [10.0, 20.0, 30.0],
        })

        evaluator = Evaluator(df, "demand", "forecast")
        metrics = evaluator.compute_metrics()

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["wape"] == 0.0


    def test_shape_after_null_removal(self):
        df = pl.DataFrame({
            "demand": [10.0, 20.0, 30.0, 40.0],
            "forecast": [None, 20.0, None, 40.0],
        })

        evaluator = Evaluator(df, "demand", "forecast")

        # dopo filtering dovrebbero restare solo 2 righe
        assert evaluator.df.height == 2