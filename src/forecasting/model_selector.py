"""Model selection utilities for choosing the best forecasting model."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


class ModelSelector:
    """Stateless utilities for selecting and ranking forecasting models by metric."""

    @staticmethod
    def select_best(metrics_df: pl.DataFrame, metric: str = "wape") -> str:
        """Return the model_name with the lowest value for *metric*.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            A Metrics_Summary_DataFrame containing at least ``model_name``
            and the requested *metric* column.
        metric : str, optional
            Column name to minimise (default ``"wape"``).

        Returns
        -------
        str
            The ``model_name`` of the best-performing model.

        Raises
        ------
        ValueError
            If *metrics_df* is empty or *metric* is not a column.
        """
        if metrics_df.is_empty():
            raise ValueError("No model results available")

        if metric not in metrics_df.columns:
            available = [c for c in metrics_df.columns if c != "model_name"]
            raise ValueError(
                f"Metric '{metric}' not found. "
                f"Available metric columns: {available}"
            )

        best_row = metrics_df.sort(metric)[0]
        model_name: str = best_row["model_name"].item()
        metric_value = best_row[metric].item()

        logger.info(
            "Selected model '%s' with %s = %s", model_name, metric, metric_value
        )

        return model_name

    @staticmethod
    def rank_models(metrics_df: pl.DataFrame, metric: str = "wape") -> pl.DataFrame:
        """Return *metrics_df* sorted ascending by *metric*.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            A Metrics_Summary_DataFrame.
        metric : str, optional
            Column name to sort by (default ``"wape"``).

        Returns
        -------
        pl.DataFrame
            The input DataFrame sorted in ascending order by *metric*.

        Raises
        ------
        ValueError
            If *metric* is not a column in *metrics_df*.
        """
        if metric not in metrics_df.columns:
            available = [c for c in metrics_df.columns if c != "model_name"]
            raise ValueError(
                f"Metric '{metric}' not found. "
                f"Available metric columns: {available}"
            )

        return metrics_df.sort(metric)
