"""Property-based tests for MultiPeriodOptimizer correctness properties."""

from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from optimization import MultiPeriodOptimizer

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for generating a small set of destination IDs
destination_ids_st = st.lists(
    st.sampled_from(["D1", "D2", "D3", "D4", "D5"]),
    min_size=1,
    max_size=5,
)

# Strategy for generating a planning horizon (1-5 consecutive dates)
planning_horizon_st = st.integers(min_value=1, max_value=5).map(
    lambda n: [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
)


@st.composite
def demand_with_duplicates(draw):
    """Generate a demand DataFrame that contains duplicate (destination_id, date) pairs.

    Returns a tuple of (demand_df, planning_horizon) where demand_df has at least
    one duplicated (destination_id, date) pair.
    """
    horizon = draw(planning_horizon_st)
    destinations = draw(
        st.lists(
            st.sampled_from(["D1", "D2", "D3", "D4", "D5"]),
            min_size=1,
            max_size=4,
        )
    )

    # Generate base rows: at least one row per (destination, date) pair
    rows = []
    for dest in destinations:
        for d in horizon:
            demand_val = draw(
                st.floats(
                    min_value=0.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            rows.append({"destination_id": dest, "date": d, "demand": demand_val})

    # Add duplicate rows for some (destination_id, date) pairs
    num_duplicates = draw(st.integers(min_value=1, max_value=max(1, len(rows))))
    for _ in range(num_duplicates):
        # Pick a random existing row to duplicate (with a different demand value)
        idx = draw(st.integers(min_value=0, max_value=len(rows) - 1))
        dup_demand = draw(
            st.floats(
                min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
            )
        )
        rows.append(
            {
                "destination_id": rows[idx]["destination_id"],
                "date": rows[idx]["date"],
                "demand": dup_demand,
            }
        )

    demand_df = pl.DataFrame(
        {
            "destination_id": [r["destination_id"] for r in rows],
            "date": [r["date"] for r in rows],
            "demand": [r["demand"] for r in rows],
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    return demand_df, horizon


# ---------------------------------------------------------------------------
# Property 6: Demand Deduplication Idempotence
# Feature: multi-period-optimization, Property 6: Demand Deduplication Idempotence
# ---------------------------------------------------------------------------


# Validates: Requirements 5.4
@settings(max_examples=100)
@given(data=demand_with_duplicates())
def test_demand_deduplication_idempotence(data):
    """Property 6: Demand Deduplication Idempotence.

    For any Demand_Time_Series containing duplicate (destination_id, date) pairs,
    preprocessing the duplicated demand SHALL produce the same result as
    preprocessing with pre-aggregated (summed) demand for each unique pair.

    **Validates: Requirements 5.4**
    """
    # Feature: multi-period-optimization, Property 6: Demand Deduplication Idempotence
    demand_with_dupes, planning_horizon = data

    # Path 1: Preprocess the demand with duplicates directly
    result_from_dupes = MultiPeriodOptimizer._preprocess_demand(
        demand_with_dupes, planning_horizon
    )

    # Path 2: Manually aggregate duplicates first, then preprocess
    pre_aggregated = demand_with_dupes.group_by("destination_id", "date").agg(
        pl.col("demand").sum()
    )
    result_from_aggregated = MultiPeriodOptimizer._preprocess_demand(
        pre_aggregated, planning_horizon
    )

    # Both results should be identical (same rows, same demand values)
    # Sort both for deterministic comparison since group_by doesn't guarantee order
    result_dupes_sorted = result_from_dupes.sort("destination_id", "date")
    result_agg_sorted = result_from_aggregated.sort("destination_id", "date")

    assert (
        result_dupes_sorted.shape == result_agg_sorted.shape
    ), f"Shape mismatch: {result_dupes_sorted.shape} vs {result_agg_sorted.shape}"
    assert result_dupes_sorted.equals(result_agg_sorted), (
        f"Results differ:\nFrom duplicates:\n{result_dupes_sorted}\n"
        f"From pre-aggregated:\n{result_agg_sorted}"
    )


# ---------------------------------------------------------------------------
# Property 8: Unreachable Destination Detection
# Feature: multi-period-optimization, Property 8: Unreachable Destination Detection
# ---------------------------------------------------------------------------


@st.composite
def unreachable_destination_problem(draw):
    """Generate a problem where some demanded destinations have no lane serving them.

    Returns a tuple of (demand_ts, origins_df, lanes_df, destinations_df,
    planning_horizon, unreachable_dest_ids).
    """
    # Generate reachable destinations (served by lanes)
    n_reachable = draw(st.integers(min_value=1, max_value=4))
    reachable_dests = [f"D{i}" for i in range(n_reachable)]

    # Generate unreachable destinations (NOT served by any lane)
    n_unreachable = draw(st.integers(min_value=1, max_value=4))
    unreachable_dests = [f"U{i}" for i in range(n_unreachable)]

    # All demanded destinations
    all_demanded = reachable_dests + unreachable_dests

    # Generate origins
    n_origins = draw(st.integers(min_value=1, max_value=3))
    origins = [f"O{i}" for i in range(n_origins)]

    # Generate planning horizon
    n_periods = draw(st.integers(min_value=1, max_value=3))
    start = date(2024, 1, 1)
    horizon = [start + timedelta(days=i) for i in range(n_periods)]

    # Build lanes: only connect origins to reachable destinations
    lane_origins = []
    lane_dests = []
    lane_costs = []
    for o in origins:
        for d in reachable_dests:
            lane_origins.append(o)
            lane_dests.append(d)
            lane_costs.append(draw(st.floats(min_value=0.1, max_value=100.0)))

    lanes_df = pl.DataFrame(
        {
            "origin_id": lane_origins,
            "destination_id": lane_dests,
            "unit_cost": lane_costs,
        }
    )

    # Build origins_df with sufficient capacity
    origins_df = pl.DataFrame(
        {
            "origin_id": origins,
            "daily_capacity": [1000.0] * n_origins,
        }
    )

    # Build demand for ALL destinations (reachable + unreachable)
    demand_dest_ids = []
    demand_dates = []
    demand_values = []
    for d in all_demanded:
        for dt in horizon:
            demand_dest_ids.append(d)
            demand_dates.append(dt)
            demand_values.append(draw(st.floats(min_value=0.1, max_value=50.0)))

    demand_ts = pl.DataFrame(
        {
            "destination_id": demand_dest_ids,
            "date": demand_dates,
            "demand": demand_values,
        }
    )

    # Build destinations_df (all demanded destinations)
    destinations_df = pl.DataFrame(
        {
            "destination_id": all_demanded,
        }
    )

    return (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        horizon,
        unreachable_dests,
    )


# Validates: Requirements 7.3
@settings(max_examples=100)
@given(problem=unreachable_destination_problem())
def test_unreachable_destination_detection(problem):
    """Property 8: Unreachable Destination Detection.

    For any set of demanded destinations where one or more have no lane serving
    them in the lanes DataFrame, the Multi_Period_Optimizer SHALL raise a ValueError
    whose message contains every unreachable destination identifier.

    **Validates: Requirements 7.3**
    """
    # Feature: multi-period-optimization, Property 8: Unreachable Destination Detection
    demand_ts, origins_df, lanes_df, destinations_df, horizon, unreachable_dests = (
        problem
    )

    optimizer = MultiPeriodOptimizer()

    try:
        optimizer.solve(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=horizon,
        )
        # If no error raised, the property is violated
        raise AssertionError(
            f"Expected ValueError for unreachable destinations {unreachable_dests}, "
            f"but solve() did not raise."
        )
    except ValueError as e:
        error_message = str(e)

        # Verify the error message mentions unreachable destinations
        assert (
            "Unreachable destinations" in error_message
        ), f"Error message does not mention 'Unreachable destinations': {error_message}"

        # Verify every unreachable destination ID appears in the error message
        for dest_id in unreachable_dests:
            assert dest_id in error_message, (
                f"Unreachable destination '{dest_id}' not found in error message: "
                f"{error_message}"
            )


# ---------------------------------------------------------------------------
# Property 7: Extra-Date Filtering
# Feature: multi-period-optimization, Property 7: Extra-Date Filtering
# ---------------------------------------------------------------------------


@st.composite
def demand_with_extra_dates(draw):
    """Generate a demand DataFrame with some dates inside and some outside the horizon.

    Returns a tuple of (demand_with_extras, demand_in_horizon_only, planning_horizon).
    """
    # Generate planning horizon
    horizon = draw(planning_horizon_st)

    # Generate at least one in-horizon row
    in_horizon_dates = draw(
        st.lists(
            st.sampled_from(horizon),
            min_size=1,
            max_size=10,
        )
    )
    in_horizon_destinations = draw(
        st.lists(
            st.sampled_from(["D1", "D2", "D3", "D4", "D5"]),
            min_size=len(in_horizon_dates),
            max_size=len(in_horizon_dates),
        )
    )
    in_horizon_demands = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=len(in_horizon_dates),
            max_size=len(in_horizon_dates),
        )
    )

    # Generate extra-date rows (dates NOT in horizon)
    horizon_set = set(horizon)
    extra_dates = draw(
        st.lists(
            st.dates(min_value=date(2019, 1, 1), max_value=date(2026, 12, 31)).filter(
                lambda d: d not in horizon_set
            ),
            min_size=1,
            max_size=5,
        )
    )
    extra_destinations = draw(
        st.lists(
            st.sampled_from(["D1", "D2", "D3", "D4", "D5"]),
            min_size=len(extra_dates),
            max_size=len(extra_dates),
        )
    )
    extra_demands = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=len(extra_dates),
            max_size=len(extra_dates),
        )
    )

    # Build the full demand DataFrame (in-horizon + extra dates)
    all_dates = in_horizon_dates + extra_dates
    all_destinations = in_horizon_destinations + extra_destinations
    all_demands = in_horizon_demands + extra_demands

    demand_full = pl.DataFrame(
        {
            "destination_id": all_destinations,
            "date": all_dates,
            "demand": all_demands,
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    # Build the in-horizon-only demand DataFrame
    demand_in_horizon = pl.DataFrame(
        {
            "destination_id": in_horizon_destinations,
            "date": in_horizon_dates,
            "demand": in_horizon_demands,
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    return demand_full, demand_in_horizon, horizon


# Validates: Requirements 5.2
@settings(max_examples=100)
@given(data=demand_with_extra_dates())
def test_extra_date_filtering(data):
    """Property 7: Extra-Date Filtering.

    For any Demand_Time_Series containing dates outside the Planning_Horizon,
    the optimization result SHALL be identical to solving with only the rows
    whose dates are within the Planning_Horizon.

    Since the LP formulation is not yet implemented, we test the
    _preprocess_demand static method directly. The property verifies that
    rows with dates outside the planning_horizon are filtered out, and the
    result is identical to preprocessing a DataFrame that only contained
    in-horizon rows to begin with.

    **Validates: Requirements 5.2**
    """
    # Feature: multi-period-optimization, Property 7: Extra-Date Filtering
    demand_full, demand_in_horizon, horizon = data

    # Preprocess the full demand (contains extra dates outside horizon)
    result_full = MultiPeriodOptimizer._preprocess_demand(demand_full, horizon)

    # Preprocess the in-horizon-only demand
    result_in_horizon = MultiPeriodOptimizer._preprocess_demand(
        demand_in_horizon, horizon
    )

    # Both results should be identical after preprocessing
    # Sort both for deterministic comparison (group_by doesn't guarantee order)
    result_full_sorted = result_full.sort(["destination_id", "date"])
    result_in_horizon_sorted = result_in_horizon.sort(["destination_id", "date"])

    assert result_full_sorted.shape == result_in_horizon_sorted.shape, (
        f"Shape mismatch: full={result_full_sorted.shape}, "
        f"in_horizon={result_in_horizon_sorted.shape}"
    )

    assert result_full_sorted.equals(result_in_horizon_sorted), (
        f"DataFrames differ after preprocessing.\n"
        f"Full (preprocessed):\n{result_full_sorted}\n"
        f"In-horizon only (preprocessed):\n{result_in_horizon_sorted}"
    )


# ---------------------------------------------------------------------------
# Property 9: Capacity Feasibility Detection
# Feature: multi-period-optimization, Property 9: Capacity Feasibility Detection
# ---------------------------------------------------------------------------


@st.composite
def capacity_infeasible_problem(draw):
    """Generate a problem where total capacity < total demand.

    Ensures all destinations ARE reachable (have lanes) so the unreachable
    check passes first, but total capacity across all origins × periods is
    strictly less than total demand.
    """
    # Generate 1-3 origins and 1-3 destinations
    n_origins = draw(st.integers(min_value=1, max_value=3))
    n_destinations = draw(st.integers(min_value=1, max_value=3))
    n_periods = draw(st.integers(min_value=1, max_value=5))

    origin_ids = [f"O{i}" for i in range(n_origins)]
    destination_ids = [f"D{i}" for i in range(n_destinations)]

    # Generate capacities (positive values)
    capacities = draw(
        st.lists(
            st.floats(
                min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
            ),
            min_size=n_origins,
            max_size=n_origins,
        )
    )

    # Compute total capacity across all origins and periods
    total_capacity = sum(capacities) * n_periods

    # Generate demand that exceeds total capacity
    excess = draw(
        st.floats(
            min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        )
    )
    total_demand_target = total_capacity + excess

    # Distribute demand evenly across destination-period pairs
    n_demand_rows = n_destinations * n_periods
    base_demand = total_demand_target / n_demand_rows

    # Build planning horizon
    start_date = date(2024, 1, 1)
    planning_horizon = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Build demand_ts: one row per destination per period
    demand_rows_dest = []
    demand_rows_date = []
    demand_rows_demand = []
    for d_id in destination_ids:
        for period_date in planning_horizon:
            demand_rows_dest.append(d_id)
            demand_rows_date.append(period_date)
            demand_rows_demand.append(base_demand)

    demand_ts = pl.DataFrame(
        {
            "destination_id": demand_rows_dest,
            "date": demand_rows_date,
            "demand": demand_rows_demand,
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    # Build origins_df
    origins_df = pl.DataFrame(
        {
            "origin_id": origin_ids,
            "daily_capacity": capacities,
        }
    )

    # Build lanes_df: connect every origin to every destination for reachability
    lane_origin_ids = []
    lane_dest_ids = []
    lane_costs = []
    for o_id in origin_ids:
        for d_id in destination_ids:
            lane_origin_ids.append(o_id)
            lane_dest_ids.append(d_id)
            lane_costs.append(
                draw(
                    st.floats(
                        min_value=0.1,
                        max_value=50.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            )

    lanes_df = pl.DataFrame(
        {
            "origin_id": lane_origin_ids,
            "destination_id": lane_dest_ids,
            "unit_cost": lane_costs,
        },
        schema={
            "origin_id": pl.Utf8,
            "destination_id": pl.Utf8,
            "unit_cost": pl.Float64,
        },
    )

    # Build destinations_df
    destinations_df = pl.DataFrame(
        {
            "destination_id": destination_ids,
        }
    )

    return demand_ts, origins_df, lanes_df, destinations_df, planning_horizon


# Validates: Requirements 7.4
@settings(max_examples=100)
@given(problem=capacity_infeasible_problem())
def test_capacity_feasibility_detection(problem):
    """Property 9: Capacity Feasibility Detection.

    For any problem where total capacity across all origins summed over all
    periods is less than total demand summed over all periods, the
    Multi_Period_Optimizer SHALL raise a ValueError whose message includes
    the total demand value, total capacity value, and the numeric shortfall.

    **Validates: Requirements 7.4**
    """
    # Feature: multi-period-optimization, Property 9: Capacity Feasibility Detection
    demand_ts, origins_df, lanes_df, destinations_df, planning_horizon = problem

    # Compute expected values
    n_periods = len(planning_horizon)
    total_capacity = origins_df["daily_capacity"].sum() * n_periods
    total_demand = demand_ts["demand"].fill_null(0).sum()
    shortfall = total_demand - total_capacity

    # Verify our precondition: total_capacity < total_demand
    assume(total_capacity < total_demand)
    assume(shortfall > 0)

    optimizer = MultiPeriodOptimizer()

    with pytest.raises(ValueError) as exc_info:
        optimizer.solve(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
        )

    error_message = str(exc_info.value)

    # Assert the error message contains the required values
    assert (
        str(total_demand) in error_message
    ), f"Error message should contain total demand ({total_demand}): {error_message}"
    assert (
        str(total_capacity) in error_message
    ), f"Error message should contain total capacity ({total_capacity}): {error_message}"
    assert (
        str(shortfall) in error_message
    ), f"Error message should contain shortfall ({shortfall}): {error_message}"


# ---------------------------------------------------------------------------
# Shared strategy: feasible_problem (reusable for Properties 1-5)
# Feature: multi-period-optimization
# ---------------------------------------------------------------------------


@st.composite
def feasible_problem(draw):
    """Generate a feasible multi-period optimization problem.

    Ensures:
    - All destinations are reachable (have at least one lane)
    - Total capacity >= total demand per period (guarantees LP feasibility)
    - All costs are non-negative, capacities are positive
    - The problem is solvable

    Returns a tuple of (demand_ts, origins_df, lanes_df, destinations_df,
    planning_horizon, initial_inventory).
    """
    # Small problem sizes for fast tests
    n_origins = draw(st.integers(min_value=1, max_value=3))
    n_destinations = draw(st.integers(min_value=1, max_value=3))
    n_periods = draw(st.integers(min_value=1, max_value=5))

    origin_ids = [f"O{i}" for i in range(n_origins)]
    destination_ids = [f"D{i}" for i in range(n_destinations)]

    # Planning horizon: consecutive dates
    start_date = date(2024, 1, 1)
    planning_horizon = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Generate capacities first so we can bound demand per period
    capacities = []
    for _ in origin_ids:
        cap = float(draw(st.integers(min_value=1, max_value=100)))
        capacities.append(cap)

    total_capacity_per_period = sum(capacities)

    # Generate demand values: ensure sum of demand across all destinations
    # in each period does not exceed total capacity per period.
    # This guarantees per-period feasibility (and thus LP feasibility).
    demand_dest_ids = []
    demand_dates = []
    demand_values = []
    for period_date in planning_horizon:
        # Max demand per destination in this period
        max_demand_per_dest = int(total_capacity_per_period / n_destinations)
        for d_id in destination_ids:
            demand_val = float(
                draw(st.integers(min_value=0, max_value=max(0, max_demand_per_dest)))
            )
            demand_dest_ids.append(d_id)
            demand_dates.append(period_date)
            demand_values.append(demand_val)

    demand_ts = pl.DataFrame(
        {
            "destination_id": demand_dest_ids,
            "date": demand_dates,
            "demand": demand_values,
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    # Generate lanes: every origin connects to every destination (ensures reachability)
    lane_origin_ids = []
    lane_dest_ids = []
    lane_costs = []
    for o_id in origin_ids:
        for d_id in destination_ids:
            lane_origin_ids.append(o_id)
            lane_dest_ids.append(d_id)
            # Use integers mapped to floats to avoid subnormal values that cause solver issues
            cost = float(draw(st.integers(min_value=0, max_value=100)))
            lane_costs.append(cost)

    lanes_df = pl.DataFrame(
        {
            "origin_id": lane_origin_ids,
            "destination_id": lane_dest_ids,
            "unit_cost": lane_costs,
        },
        schema={
            "origin_id": pl.Utf8,
            "destination_id": pl.Utf8,
            "unit_cost": pl.Float64,
        },
    )

    origins_df = pl.DataFrame(
        {
            "origin_id": origin_ids,
            "daily_capacity": capacities,
        }
    )

    # Destinations DataFrame (optionally with holding_cost)
    include_holding_cost = draw(st.booleans())
    if include_holding_cost:
        holding_costs = [
            float(draw(st.integers(min_value=0, max_value=10))) for _ in destination_ids
        ]
        destinations_df = pl.DataFrame(
            {
                "destination_id": destination_ids,
                "holding_cost": holding_costs,
            }
        )
    else:
        destinations_df = pl.DataFrame(
            {
                "destination_id": destination_ids,
            }
        )

    # Optional initial inventory (non-negative)
    include_initial_inv = draw(st.booleans())
    if include_initial_inv:
        initial_inventory = {}
        for d_id in destination_ids:
            inv_val = float(draw(st.integers(min_value=0, max_value=20)))
            initial_inventory[d_id] = inv_val
    else:
        initial_inventory = None

    return (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    )


# ---------------------------------------------------------------------------
# Property 1: Inventory Balance Invariant
# Feature: multi-period-optimization, Property 1: Inventory Balance Invariant
# ---------------------------------------------------------------------------


# Validates: Requirements 2.1, 2.2, 2.3
@settings(max_examples=100)
@given(problem=feasible_problem())
def test_inventory_balance_invariant(problem):
    """Property 1: Inventory Balance Invariant.

    For any feasible multi-period optimization problem with any valid initial
    inventory (including the default of zero), the solved inventory values SHALL
    satisfy `inventory[d, t] = inventory[d, t-1] + sum(flow[o, d, t] for all o)
    - demand[d, t]` for all destinations `d` and all periods `t`, within
    floating-point tolerance (1e-6).

    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    # Feature: multi-period-optimization, Property 1: Inventory Balance Invariant
    (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    ) = problem

    optimizer = MultiPeriodOptimizer()
    result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    # Build demand lookup: (destination_id, date) -> demand value
    demand_map: dict[tuple[str, date], float] = {}
    for row in demand_ts.to_dicts():
        key = (row["destination_id"], row["date"])
        # Sum duplicates (matching preprocessing behavior)
        demand_map[key] = demand_map.get(key, 0.0) + (
            row["demand"] if row["demand"] is not None else 0.0
        )

    # Build inventory lookup: (destination_id, period) -> inventory value
    inv_map: dict[tuple[str, date], float] = {}
    for row in result.inventory.to_dicts():
        inv_map[(row["destination_id"], row["period"])] = row["inventory"]

    # Build flow lookup: (destination_id, period) -> total inflow
    inflow_map: dict[tuple[str, date], float] = {}
    for row in result.flows.to_dicts():
        key = (row["destination_id"], row["period"])
        inflow_map[key] = inflow_map.get(key, 0.0) + row["flow"]

    # Resolve initial inventory
    init_inv = initial_inventory if initial_inventory is not None else {}

    # Get all destinations from the inventory result
    destinations = sorted(result.inventory["destination_id"].unique().to_list())

    tolerance = 1e-6

    for d_id in destinations:
        for t_idx, t in enumerate(planning_horizon):
            # Get inventory at this period
            actual_inv = inv_map.get((d_id, t), 0.0)

            # Get previous inventory
            if t_idx == 0:
                prev_inv = init_inv.get(d_id, 0.0)
            else:
                prev_t = planning_horizon[t_idx - 1]
                prev_inv = inv_map.get((d_id, prev_t), 0.0)

            # Get inflow for this destination at this period
            inflow = inflow_map.get((d_id, t), 0.0)

            # Get demand for this destination at this period
            demand_val = demand_map.get((d_id, t), 0.0)

            # Expected inventory: prev_inv + inflow - demand
            expected_inv = prev_inv + inflow - demand_val

            assert abs(actual_inv - expected_inv) <= tolerance, (
                f"Inventory balance violated for destination '{d_id}' at period {t}:\n"
                f"  actual inventory = {actual_inv}\n"
                f"  expected = prev_inv ({prev_inv}) + inflow ({inflow}) - demand ({demand_val}) = {expected_inv}\n"
                f"  difference = {abs(actual_inv - expected_inv)}"
            )


# ---------------------------------------------------------------------------
# Shared strategy: feasible_problem
# ---------------------------------------------------------------------------


@st.composite
def feasible_problem(draw):
    """Generate a feasible multi-period optimization problem.

    Ensures:
    - All destinations are reachable (have at least one lane serving them)
    - Total capacity across all origins × periods >= total demand
    - All costs are non-negative, all capacities are positive

    Returns a tuple of (demand_ts, origins_df, lanes_df, destinations_df,
    planning_horizon, initial_inventory).
    """
    # Generate dimensions
    n_origins = draw(st.integers(min_value=1, max_value=3))
    n_destinations = draw(st.integers(min_value=1, max_value=3))
    n_periods = draw(st.integers(min_value=1, max_value=4))

    origin_ids = [f"O{i}" for i in range(n_origins)]
    destination_ids = [f"D{i}" for i in range(n_destinations)]

    # Generate planning horizon
    start = date(2024, 1, 1)
    planning_horizon = [start + timedelta(days=i) for i in range(n_periods)]

    # Generate capacities first (positive values)
    capacities = []
    for _ in range(n_origins):
        cap = draw(
            st.floats(
                min_value=1.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            )
        )
        capacities.append(cap)

    total_capacity_per_period = sum(capacities)

    # Generate demand values that are feasible per-period:
    # sum of demand across all destinations in any single period must not exceed
    # total capacity per period (sum of all origin capacities)
    demand_dest_ids = []
    demand_dates = []
    demand_values = []
    for dt in planning_horizon:
        # For each period, generate demands that sum to at most total_capacity_per_period
        period_demands = []
        for d_id in destination_ids:
            remaining_budget = total_capacity_per_period - sum(period_demands)
            max_demand = min(50.0, remaining_budget)
            if max_demand < 0:
                max_demand = 0.0
            demand_val = draw(
                st.floats(
                    min_value=0.0,
                    max_value=max(0.0, max_demand),
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                )
            )
            period_demands.append(demand_val)
            demand_dest_ids.append(d_id)
            demand_dates.append(dt)
            demand_values.append(demand_val)

    demand_ts = pl.DataFrame(
        {
            "destination_id": demand_dest_ids,
            "date": demand_dates,
            "demand": demand_values,
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    origins_df = pl.DataFrame(
        {
            "origin_id": origin_ids,
            "daily_capacity": capacities,
        }
    )

    # Generate lanes: connect every origin to every destination (ensures reachability)
    lane_origin_ids = []
    lane_dest_ids = []
    lane_costs = []
    for o_id in origin_ids:
        for d_id in destination_ids:
            lane_origin_ids.append(o_id)
            lane_dest_ids.append(d_id)
            cost = draw(
                st.floats(
                    min_value=0.1,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                )
            )
            lane_costs.append(cost)

    lanes_df = pl.DataFrame(
        {
            "origin_id": lane_origin_ids,
            "destination_id": lane_dest_ids,
            "unit_cost": lane_costs,
        },
        schema={
            "origin_id": pl.Utf8,
            "destination_id": pl.Utf8,
            "unit_cost": pl.Float64,
        },
    )

    # Build destinations_df (optionally with holding_cost)
    include_holding_cost = draw(st.booleans())
    if include_holding_cost:
        holding_costs = []
        for _ in destination_ids:
            h_cost = draw(
                st.floats(
                    min_value=0.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                )
            )
            holding_costs.append(h_cost)
        destinations_df = pl.DataFrame(
            {
                "destination_id": destination_ids,
                "holding_cost": holding_costs,
            }
        )
    else:
        destinations_df = pl.DataFrame(
            {
                "destination_id": destination_ids,
            }
        )

    # Generate optional initial inventory (non-negative)
    use_initial_inventory = draw(st.booleans())
    if use_initial_inventory:
        initial_inventory = {}
        for d_id in destination_ids:
            inv_val = draw(
                st.floats(
                    min_value=0.0,
                    max_value=20.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                )
            )
            initial_inventory[d_id] = inv_val
    else:
        initial_inventory = None

    return (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    )


# ---------------------------------------------------------------------------
# Property 2: Capacity Constraint Invariant
# Feature: multi-period-optimization, Property 2: Capacity Constraint Invariant
# ---------------------------------------------------------------------------


# Validates: Requirements 3.1, 3.2
@settings(max_examples=100)
@given(problem=feasible_problem())
def test_capacity_constraint_invariant(problem):
    """Property 2: Capacity Constraint Invariant.

    For any feasible multi-period optimization problem, the sum of all solved
    flow values from any origin `o` in any single period `t` SHALL not exceed
    the `daily_capacity` of that origin, within floating-point tolerance (1e-6).

    **Validates: Requirements 3.1, 3.2**
    """
    # Feature: multi-period-optimization, Property 2: Capacity Constraint Invariant
    (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    ) = problem

    optimizer = MultiPeriodOptimizer()
    result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    # Build capacity lookup: origin_id -> daily_capacity
    capacity_map = dict(
        zip(
            origins_df["origin_id"].to_list(),
            origins_df["daily_capacity"].to_list(),
        )
    )

    # For each origin and period, sum all flows and check against capacity
    flows_df = result.flows

    if flows_df.is_empty():
        # No flows means all capacity constraints are trivially satisfied
        return

    for origin_id, daily_capacity in capacity_map.items():
        for t in planning_horizon:
            # Sum all flows from this origin in this period
            origin_period_flows = flows_df.filter(
                (pl.col("origin_id") == origin_id) & (pl.col("period") == t)
            )
            total_flow = (
                origin_period_flows["flow"].sum()
                if not origin_period_flows.is_empty()
                else 0.0
            )

            assert total_flow <= daily_capacity + 1e-6, (
                f"Capacity constraint violated for origin '{origin_id}' at period {t}: "
                f"total flow = {total_flow}, daily_capacity = {daily_capacity}"
            )


# ---------------------------------------------------------------------------
# Shared strategy: feasible_problem
# ---------------------------------------------------------------------------


@st.composite
def feasible_problem(draw):
    """Generate a feasible multi-period optimization problem.

    Ensures:
    - All destinations are reachable (have at least one lane serving them)
    - Per-period total capacity (sum of all origins) >= demand in that period
    - All costs are non-negative and numerically well-behaved
    - All capacities are positive

    Returns a tuple of (demand_ts, origins_df, lanes_df, destinations_df,
    planning_horizon, initial_inventory).
    """
    # Generate dimensions
    n_origins = draw(st.integers(min_value=1, max_value=3))
    n_destinations = draw(st.integers(min_value=1, max_value=3))
    n_periods = draw(st.integers(min_value=1, max_value=4))

    origin_ids = [f"O{i}" for i in range(n_origins)]
    destination_ids = [f"D{i}" for i in range(n_destinations)]

    # Generate planning horizon
    start_date = date(2024, 1, 1)
    planning_horizon = [start_date + timedelta(days=i) for i in range(n_periods)]

    # Generate demand values (non-negative)
    # Structure: demand_values[d_idx * n_periods + t_idx]
    demand_dest_ids = []
    demand_dates = []
    demand_values = []
    for d_id in destination_ids:
        for dt in planning_horizon:
            demand_dest_ids.append(d_id)
            demand_dates.append(dt)
            demand_values.append(
                draw(
                    st.floats(
                        min_value=0.0,
                        max_value=50.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            )

    demand_ts = pl.DataFrame(
        {
            "destination_id": demand_dest_ids,
            "date": demand_dates,
            "demand": demand_values,
        },
        schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
    )

    # Compute max demand in any single period across all destinations
    # demand_values layout: [D0_t0, D0_t1, ..., D1_t0, D1_t1, ...]
    max_period_demand = 0.0
    for t_idx in range(n_periods):
        period_demand = sum(
            demand_values[d_idx * n_periods + t_idx] for d_idx in range(n_destinations)
        )
        max_period_demand = max(max_period_demand, period_demand)

    # Generate lanes: connect every origin to every destination (ensures reachability)
    # Use min_value=0.1 to avoid denormalized floats that cause solver ABNORMAL status
    lane_origins = []
    lane_dests = []
    lane_costs = []
    for o_id in origin_ids:
        for d_id in destination_ids:
            lane_origins.append(o_id)
            lane_dests.append(d_id)
            lane_costs.append(
                draw(
                    st.floats(
                        min_value=0.1,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
            )

    lanes_df = pl.DataFrame(
        {
            "origin_id": lane_origins,
            "destination_id": lane_dests,
            "unit_cost": lane_costs,
        },
        schema={
            "origin_id": pl.Utf8,
            "destination_id": pl.Utf8,
            "unit_cost": pl.Float64,
        },
    )

    # Generate capacities: ensure per-period total capacity >= max period demand
    # Each origin gets enough capacity so that sum of all origins >= max_period_demand
    min_capacity_per_origin = (max_period_demand / n_origins) + 1.0
    capacities = []
    for _ in range(n_origins):
        cap = draw(
            st.floats(
                min_value=max(min_capacity_per_origin, 1.0),
                max_value=max(min_capacity_per_origin, 1.0) + 100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        capacities.append(cap)

    origins_df = pl.DataFrame(
        {
            "origin_id": origin_ids,
            "daily_capacity": capacities,
        }
    )

    # Optionally include holding costs (use reasonable range to avoid numerical issues)
    include_holding_cost = draw(st.booleans())
    if include_holding_cost:
        holding_costs = [
            draw(
                st.floats(
                    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
                )
            )
            for _ in destination_ids
        ]
        destinations_df = pl.DataFrame(
            {
                "destination_id": destination_ids,
                "holding_cost": holding_costs,
            }
        )
    else:
        destinations_df = pl.DataFrame(
            {
                "destination_id": destination_ids,
            }
        )

    # No initial inventory — avoids infeasibility edge cases
    initial_inventory = None

    return (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    )


# ---------------------------------------------------------------------------
# Property 3: Non-Negativity Invariant
# Feature: multi-period-optimization, Property 3: Non-Negativity Invariant
# ---------------------------------------------------------------------------


# Validates: Requirements 1.3
@settings(max_examples=100)
@given(problem=feasible_problem())
def test_non_negativity_invariant(problem):
    """Property 3: Non-Negativity Invariant.

    For any feasible multi-period optimization problem, all solved flow values
    and all solved inventory values SHALL be greater than or equal to zero.

    **Validates: Requirements 1.3**
    """
    # Feature: multi-period-optimization, Property 3: Non-Negativity Invariant
    (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    ) = problem

    optimizer = MultiPeriodOptimizer()
    result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    # Assert all flow values are non-negative
    if not result.flows.is_empty():
        flow_values = result.flows["flow"].to_list()
        for val in flow_values:
            assert val >= 0, f"Negative flow value found: {val}"

    # Assert all inventory values are non-negative
    if not result.inventory.is_empty():
        inventory_values = result.inventory["inventory"].to_list()
        for val in inventory_values:
            assert val >= 0, f"Negative inventory value found: {val}"


# ---------------------------------------------------------------------------
# Property 4: Objective Value Consistency
# Feature: multi-period-optimization, Property 4: Objective Value Consistency
# ---------------------------------------------------------------------------


# Validates: Requirements 4.2, 4.3, 6.3
@settings(max_examples=100)
@given(problem=feasible_problem())
def test_objective_value_consistency(problem):
    """Property 4: Objective Value Consistency.

    For any feasible multi-period optimization problem, the returned `total_cost`
    SHALL equal the sum of `unit_cost[o,d] × flow[o,d,t]` across all lanes and
    periods, plus the sum of `holding_cost[d] × inventory[d,t]` across all
    destinations and periods (where holding_cost is zero if not provided), within
    floating-point tolerance (1e-6).

    **Validates: Requirements 4.2, 4.3, 6.3**
    """
    # Feature: multi-period-optimization, Property 4: Objective Value Consistency
    (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    ) = problem

    optimizer = MultiPeriodOptimizer()
    result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    # Build unit_cost lookup: (origin_id, destination_id) -> unit_cost
    cost_map: dict[tuple[str, str], float] = {}
    for row in lanes_df.to_dicts():
        cost_map[(row["origin_id"], row["destination_id"])] = row["unit_cost"]

    # Compute transportation cost from flows
    transport_cost = 0.0
    for row in result.flows.to_dicts():
        lane_key = (row["origin_id"], row["destination_id"])
        unit_cost = cost_map[lane_key]
        transport_cost += unit_cost * row["flow"]

    # Compute holding cost from inventory
    holding_cost_total = 0.0
    has_holding_cost = "holding_cost" in destinations_df.columns
    if has_holding_cost:
        # Build holding_cost lookup: destination_id -> holding_cost
        holding_cost_map: dict[str, float] = dict(
            zip(
                destinations_df["destination_id"].to_list(),
                destinations_df["holding_cost"].to_list(),
            )
        )
        for row in result.inventory.to_dicts():
            h_cost = holding_cost_map.get(row["destination_id"], 0.0)
            holding_cost_total += h_cost * row["inventory"]

    # Expected total cost
    computed_cost = transport_cost + holding_cost_total

    # The flows DataFrame only includes flows > 1e-6, so the solver's objective
    # may include small contributions from filtered flows. We account for this by
    # computing an upper bound on the cost of filtered flows:
    # each filtered lane-period has flow <= 1e-6, and there are at most
    # n_lanes * n_periods such entries, each contributing at most max_cost * 1e-6.
    n_lanes = len(lanes_df)
    n_periods = len(planning_horizon)
    max_unit_cost = lanes_df["unit_cost"].max() if not lanes_df.is_empty() else 0.0
    max_filtered_cost = n_lanes * n_periods * max_unit_cost * 1e-6

    # Use tolerance that accounts for filtered flow contributions
    tolerance = max(1e-6, max_filtered_cost)

    assert abs(result.total_cost - computed_cost) <= tolerance, (
        f"Objective value inconsistency:\n"
        f"  result.total_cost = {result.total_cost}\n"
        f"  computed_cost = {computed_cost}\n"
        f"  transport_cost = {transport_cost}\n"
        f"  holding_cost = {holding_cost_total}\n"
        f"  difference = {abs(result.total_cost - computed_cost)}\n"
        f"  tolerance = {tolerance}"
    )

    # Also verify the breakdown fields on the result
    assert abs(result.transportation_cost - transport_cost) <= tolerance, (
        f"transportation_cost mismatch:\n"
        f"  result.transportation_cost = {result.transportation_cost}\n"
        f"  computed transport_cost = {transport_cost}\n"
        f"  difference = {abs(result.transportation_cost - transport_cost)}"
    )
    assert abs(result.holding_cost - holding_cost_total) <= 1e-6, (
        f"holding_cost mismatch:\n"
        f"  result.holding_cost = {result.holding_cost}\n"
        f"  computed holding_cost_total = {holding_cost_total}\n"
        f"  difference = {abs(result.holding_cost - holding_cost_total)}"
    )
    assert result.transportation_cost + result.holding_cost == pytest.approx(
        result.total_cost, abs=tolerance
    ), (
        f"transportation_cost + holding_cost != total_cost:\n"
        f"  {result.transportation_cost} + {result.holding_cost} = "
        f"{result.transportation_cost + result.holding_cost} vs {result.total_cost}"
    )


# ---------------------------------------------------------------------------
# Property 5: Output Completeness
# Feature: multi-period-optimization, Property 5: Output Completeness
# ---------------------------------------------------------------------------


# Validates: Requirements 6.1, 6.2, 6.4
@settings(max_examples=100)
@given(problem=feasible_problem())
def test_output_completeness(problem):
    """Property 5: Output Completeness.

    For any feasible multi-period optimization problem with `D` demanded
    destinations and `T` periods, the inventory DataFrame SHALL contain exactly
    `D × T` rows, and all flow values in the flows DataFrame SHALL exceed 1e-6,
    and both DataFrames SHALL conform to their defined schemas.

    **Validates: Requirements 6.1, 6.2, 6.4**
    """
    # Feature: multi-period-optimization, Property 5: Output Completeness
    (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    ) = problem

    optimizer = MultiPeriodOptimizer()
    result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    # Determine D (number of unique demanded destinations) and T (number of periods)
    n_destinations = demand_ts["destination_id"].n_unique()
    n_periods = len(planning_horizon)

    # 1. Assert inventory has exactly D × T rows
    expected_inventory_rows = n_destinations * n_periods
    assert result.inventory.shape[0] == expected_inventory_rows, (
        f"Inventory row count mismatch: expected {expected_inventory_rows} "
        f"(D={n_destinations} × T={n_periods}), got {result.inventory.shape[0]}"
    )

    # 2. Assert all flow values exceed 1e-6
    if not result.flows.is_empty():
        min_flow = result.flows["flow"].min()
        assert (
            min_flow > 1e-6
        ), f"Flow value below threshold: min flow = {min_flow}, expected > 1e-6"

    # 3. Assert flows schema is [origin_id: Utf8, destination_id: Utf8, period: Date, flow: Float64]
    expected_flows_schema = {
        "origin_id": pl.Utf8,
        "destination_id": pl.Utf8,
        "period": pl.Date,
        "flow": pl.Float64,
    }
    actual_flows_schema = dict(result.flows.schema)
    assert (
        actual_flows_schema == expected_flows_schema
    ), f"Flows schema mismatch:\n  expected: {expected_flows_schema}\n  actual: {actual_flows_schema}"

    # 4. Assert inventory schema is [destination_id: Utf8, period: Date, inventory: Float64]
    expected_inventory_schema = {
        "destination_id": pl.Utf8,
        "period": pl.Date,
        "inventory": pl.Float64,
    }
    actual_inventory_schema = dict(result.inventory.schema)
    assert (
        actual_inventory_schema == expected_inventory_schema
    ), f"Inventory schema mismatch:\n  expected: {expected_inventory_schema}\n  actual: {actual_inventory_schema}"


# ---------------------------------------------------------------------------
# Property 6: Lead Time Equivalence
# Feature: multi-period-lead-times, Property 6: Backward Compatibility
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(problem=feasible_problem())
def test_lead_time_zero_equivalence(problem):
    """Property 6: Backward Compatibility with Zero Lead Time.

    When lanes explicitly specify `lead_time_days = 0`, the optimizer output
    SHALL be identical to when `lead_time_days` column is absent.
    """
    (
        demand_ts,
        origins_df,
        lanes_df,
        destinations_df,
        planning_horizon,
        initial_inventory,
    ) = problem

    optimizer = MultiPeriodOptimizer()

    result_without_lt = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    lanes_with_zero_lt = lanes_df.with_columns(
        pl.lit(0).alias("lead_time_days").cast(pl.Int64)
    )

    result_with_zero_lt = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_with_zero_lt,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
        initial_inventory=initial_inventory,
    )

    assert result_with_zero_lt.total_cost == pytest.approx(
        result_without_lt.total_cost, abs=1e-4
    )
