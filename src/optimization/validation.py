"""Validation functions for the transportation optimization module.

Shared utilities (validate_not_empty, validate_columns, etc.) can be reused
across the module. LP-specific orchestrators (validate_inputs, check_feasibility)
are called by MultiPeriodOptimizer.solve() before building the LP.

All functions raise ValueError with a descriptive message on the first
violation found.
"""

from datetime import date

import polars as pl

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def validate_not_empty(**named_dfs: pl.DataFrame) -> None:
    """Raise ValueError if any named DataFrame is empty."""
    for name, df in named_dfs.items():
        if df.is_empty():
            raise ValueError(name)


def validate_columns(
    df: pl.DataFrame,
    required: set[str],
    df_name: str,
    *,
    message_template: str | None = None,
) -> None:
    """Raise ValueError if a DataFrame is missing required columns."""
    missing = required - set(df.columns)
    if missing:
        if message_template is not None:
            raise ValueError(
                message_template.format(
                    df_name=df_name,
                    missing=sorted(missing),
                    required=sorted(required),
                )
            )
        raise ValueError(
            f"{df_name} missing columns {sorted(missing)}. "
            f"Expected: {sorted(required)}"
        )


def check_unreachable_destinations(
    demand_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    *,
    demand_col: str = "destination_id",
    lanes_col: str = "destination_id",
    message_template: str = "Unreachable destinations (no lane available): {unreachable}",
) -> None:
    """Raise ValueError if demanded destinations have no serving lane."""
    demanded_ids = set(demand_df[demand_col].unique().to_list())
    lane_dest_ids = set(lanes_df[lanes_col].unique().to_list())
    unreachable = sorted(demanded_ids - lane_dest_ids)
    if unreachable:
        raise ValueError(message_template.format(unreachable=unreachable))


def check_capacity_feasibility(
    total_demand: float,
    total_capacity: float,
    *,
    message_template: str = (
        "Insufficient total daily_capacity. "
        "Total demand: {total_demand}, "
        "total daily_capacity: {total_capacity}, "
        "shortfall: {shortfall}"
    ),
) -> None:
    """Raise ValueError if total capacity is less than total demand."""
    if total_capacity < total_demand:
        shortfall = total_demand - total_capacity
        raise ValueError(
            message_template.format(
                total_demand=total_demand,
                total_capacity=total_capacity,
                shortfall=shortfall,
            )
        )


def validate_non_negative_costs(
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame | None = None,
) -> None:
    """Raise ValueError if any cost column contains negative values."""
    if "unit_cost" in lanes_df.columns:
        negative_costs = lanes_df.filter(pl.col("unit_cost") < 0)
        if not negative_costs.is_empty():
            invalid_rows = negative_costs.select(
                "origin_id", "destination_id", "unit_cost"
            ).to_dicts()
            raise ValueError(f"Negative unit_cost values found: {invalid_rows}")

    if destinations_df is not None and "holding_cost" in destinations_df.columns:
        negative_holding = destinations_df.filter(pl.col("holding_cost") < 0)
        if not negative_holding.is_empty():
            invalid_rows = negative_holding.select(
                "destination_id", "holding_cost"
            ).to_dicts()
            raise ValueError(f"Negative holding_cost values found: {invalid_rows}")


def validate_non_negative_lead_times(lanes_df: pl.DataFrame) -> None:
    """Raise ValueError if any lead_time_days column contains negative values."""
    if "lead_time_days" in lanes_df.columns:
        negative_lt = lanes_df.filter(pl.col("lead_time_days") < 0)
        if not negative_lt.is_empty():
            invalid_rows = negative_lt.select(
                "origin_id", "destination_id", "lead_time_days"
            ).to_dicts()
            raise ValueError(f"Negative lead_time_days values found: {invalid_rows}")


def validate_positive_capacities(origins_df: pl.DataFrame) -> None:
    """Raise ValueError if any origin has non-positive daily_capacity."""
    if "daily_capacity" in origins_df.columns:
        invalid = origins_df.filter(pl.col("daily_capacity") <= 0)
        if not invalid.is_empty():
            invalid_rows = invalid.select("origin_id", "daily_capacity").to_dicts()
            raise ValueError(
                f"Non-positive daily_capacity values found: {invalid_rows}"
            )


def validate_origins_in_lanes(origins_df: pl.DataFrame, lanes_df: pl.DataFrame) -> None:
    """Raise ValueError if lanes reference origins not in origins_df."""
    origin_ids = set(origins_df["origin_id"].to_list())
    lane_origin_ids = set(lanes_df["origin_id"].to_list())
    missing_origins = sorted(lane_origin_ids - origin_ids)
    if missing_origins:
        raise ValueError(
            f"Origins referenced in lanes but missing from origins_df: "
            f"{missing_origins}"
        )


# ---------------------------------------------------------------------------
# LP-specific validation
# ---------------------------------------------------------------------------

MAX_VARIABLES = 1_000_000
"""Default upper bound on the total LP decision-variable count.

Equal to ``n_lanes × n_periods + n_destinations × n_periods``. Problems
exceeding this limit are rejected before LP construction to prevent
out-of-memory conditions. Override via the ``max_variables`` parameter on
:class:`~optimization.optimizer.MultiPeriodOptimizer`.
"""


def validate_inputs(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame,
    planning_horizon: list[date],
    initial_inventory: dict[str, float] | None,
    max_variables: int = MAX_VARIABLES,
) -> None:
    """Run structural validation. Raises ValueError on the first violation.

    Parameters
    ----------
    demand_ts, origins_df, lanes_df, destinations_df, planning_horizon,
    initial_inventory
        Standard optimizer inputs.
    max_variables : int
        Maximum allowed total decision-variable count
        (``flow_vars + inventory_vars``). Defaults to ``MAX_VARIABLES``
        (1 000 000). Pass a smaller value to enforce tighter memory limits,
        or a larger value for large-scale problems on capable hardware.
    """
    _validate_not_empty(demand_ts, origins_df, lanes_df, planning_horizon)
    validate_columns(
        demand_ts,
        {"destination_id", "date", "demand"},
        "Demand time series",
        message_template="{df_name} missing required columns: {missing}",
    )
    validate_non_negative_costs(lanes_df, destinations_df)
    validate_non_negative_lead_times(lanes_df)
    validate_positive_capacities(origins_df)
    validate_origins_in_lanes(origins_df, lanes_df)
    _validate_initial_inventory(initial_inventory)
    _validate_variable_count(lanes_df, demand_ts, planning_horizon, max_variables)


def check_feasibility(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    """Raise ValueError if the problem is structurally infeasible before solving."""
    check_unreachable_destinations(
        demand_ts,
        lanes_df,
        message_template="Unreachable destinations (no lane serves them): {unreachable}",
    )
    _check_capacity_feasibility(demand_ts, origins_df, planning_horizon)


def _validate_not_empty(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    validate_not_empty(
        **{
            "no demand data available": demand_ts,
            "Origins DataFrame is empty": origins_df,
            "Lanes DataFrame is empty": lanes_df,
        }
    )
    if len(planning_horizon) == 0:
        raise ValueError("Planning horizon contains zero periods")


def _validate_initial_inventory(initial_inventory: dict[str, float] | None) -> None:
    if initial_inventory is None:
        return
    for dest_id, value in initial_inventory.items():
        if value < 0:
            raise ValueError(
                f"Negative initial inventory for destination '{dest_id}': {value}"
            )


def _validate_variable_count(
    lanes_df: pl.DataFrame,
    demand_ts: pl.DataFrame,
    planning_horizon: list[date],
    max_variables: int = MAX_VARIABLES,
) -> None:
    n_periods = len(planning_horizon)
    n_lanes = len(lanes_df)
    n_destinations = demand_ts["destination_id"].n_unique()

    flow_vars = n_lanes * n_periods
    inventory_vars = n_destinations * n_periods
    total_vars = flow_vars + inventory_vars

    if total_vars > max_variables:
        raise ValueError(
            f"Variable count ({total_vars}) exceeds maximum limit "
            f"({max_variables}). "
            f"Flow variables: {flow_vars}, "
            f"inventory variables: {inventory_vars}."
        )


def _check_capacity_feasibility(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    n_periods = len(planning_horizon)
    total_capacity = origins_df["daily_capacity"].sum() * n_periods
    total_demand = demand_ts["demand"].fill_null(0).sum()

    check_capacity_feasibility(
        total_demand,
        total_capacity,
        message_template=(
            "Insufficient total capacity: total demand = {total_demand}, "
            "total capacity = {total_capacity}, shortfall = {shortfall}"
        ),
    )
