"""Unit tests for MultiPeriodOptimizer input validation."""

from datetime import date

import polars as pl
import pytest

from optimization import MultiPeriodOptimizer

# ---------------------------------------------------------------------------
# Fixtures: minimal valid inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_demand_ts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "destination_id": ["D1", "D1", "D2", "D2"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
            "demand": [10.0, 15.0, 20.0, 25.0],
        }
    )


@pytest.fixture
def valid_origins_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_id": ["O1", "O2"],
            "daily_capacity": [100.0, 200.0],
        }
    )


@pytest.fixture
def valid_lanes_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_id": ["O1", "O2"],
            "destination_id": ["D1", "D2"],
            "unit_cost": [5.0, 3.0],
        }
    )


@pytest.fixture
def valid_destinations_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "destination_id": ["D1", "D2"],
            "holding_cost": [1.0, 2.0],
        }
    )


@pytest.fixture
def valid_planning_horizon() -> list[date]:
    return [date(2024, 1, 1), date(2024, 1, 2)]


# ---------------------------------------------------------------------------
# Tests: Empty inputs raise ValueError
# ---------------------------------------------------------------------------


class TestEmptyInputValidation:
    """Test empty demand, origins, lanes, planning_horizon raise ValueError."""

    def test_empty_demand_raises(
        self,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        empty_demand = pl.DataFrame(
            {"destination_id": [], "date": [], "demand": []},
            schema={"destination_id": pl.Utf8, "date": pl.Date, "demand": pl.Float64},
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="no demand data available"):
            optimizer.solve(
                empty_demand,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_empty_origins_raises(
        self,
        valid_demand_ts,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        empty_origins = pl.DataFrame(
            {"origin_id": [], "daily_capacity": []},
            schema={"origin_id": pl.Utf8, "daily_capacity": pl.Float64},
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="Origins DataFrame is empty"):
            optimizer.solve(
                valid_demand_ts,
                empty_origins,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_empty_lanes_raises(
        self,
        valid_demand_ts,
        valid_origins_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        empty_lanes = pl.DataFrame(
            {"origin_id": [], "destination_id": [], "unit_cost": []},
            schema={
                "origin_id": pl.Utf8,
                "destination_id": pl.Utf8,
                "unit_cost": pl.Float64,
            },
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="Lanes DataFrame is empty"):
            optimizer.solve(
                valid_demand_ts,
                valid_origins_df,
                empty_lanes,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_empty_planning_horizon_raises(
        self, valid_demand_ts, valid_origins_df, valid_lanes_df, valid_destinations_df
    ):
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="Planning horizon contains zero periods"):
            optimizer.solve(
                valid_demand_ts,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                [],
            )


# ---------------------------------------------------------------------------
# Tests: Missing schema columns raise ValueError
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Test missing schema columns raise ValueError."""

    def test_missing_destination_id_column(
        self,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_demand = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "demand": [10.0],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Demand time series missing required columns"
        ):
            optimizer.solve(
                bad_demand,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_missing_date_column(
        self,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_demand = pl.DataFrame(
            {
                "destination_id": ["D1"],
                "demand": [10.0],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Demand time series missing required columns"
        ):
            optimizer.solve(
                bad_demand,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_missing_demand_column(
        self,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_demand = pl.DataFrame(
            {
                "destination_id": ["D1"],
                "date": [date(2024, 1, 1)],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Demand time series missing required columns"
        ):
            optimizer.solve(
                bad_demand,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_missing_multiple_columns(
        self,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_demand = pl.DataFrame({"some_col": [1.0]})
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Demand time series missing required columns"
        ):
            optimizer.solve(
                bad_demand,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )


# ---------------------------------------------------------------------------
# Tests: Negative costs, non-positive capacities, negative initial inventory
# ---------------------------------------------------------------------------


class TestValueValidation:
    """Test negative costs, non-positive capacities, negative initial inventory."""

    def test_negative_unit_cost_raises(
        self,
        valid_demand_ts,
        valid_origins_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_lanes = pl.DataFrame(
            {
                "origin_id": ["O1", "O2"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [-1.0, 3.0],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="Negative unit_cost values found"):
            optimizer.solve(
                valid_demand_ts,
                valid_origins_df,
                bad_lanes,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_negative_holding_cost_raises(
        self, valid_demand_ts, valid_origins_df, valid_lanes_df, valid_planning_horizon
    ):
        bad_destinations = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "holding_cost": [1.0, -0.5],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="Negative holding_cost values found"):
            optimizer.solve(
                valid_demand_ts,
                valid_origins_df,
                valid_lanes_df,
                bad_destinations,
                valid_planning_horizon,
            )

    def test_zero_capacity_raises(
        self,
        valid_demand_ts,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_origins = pl.DataFrame(
            {
                "origin_id": ["O1", "O2"],
                "daily_capacity": [0.0, 200.0],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Non-positive daily_capacity values found"
        ):
            optimizer.solve(
                valid_demand_ts,
                bad_origins,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_negative_capacity_raises(
        self,
        valid_demand_ts,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        bad_origins = pl.DataFrame(
            {
                "origin_id": ["O1", "O2"],
                "daily_capacity": [-50.0, 200.0],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Non-positive daily_capacity values found"
        ):
            optimizer.solve(
                valid_demand_ts,
                bad_origins,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )

    def test_negative_initial_inventory_raises(
        self,
        valid_demand_ts,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Negative initial inventory for destination 'D1'"
        ):
            optimizer.solve(
                valid_demand_ts,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
                initial_inventory={"D1": -10.0, "D2": 5.0},
            )

    def test_origins_in_lanes_missing_from_origins_df(
        self, valid_demand_ts, valid_destinations_df, valid_planning_horizon
    ):
        origins_df = pl.DataFrame(
            {
                "origin_id": ["O1"],
                "daily_capacity": [100.0],
            }
        )
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O3"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [5.0, 3.0],
            }
        )
        optimizer = MultiPeriodOptimizer()
        with pytest.raises(
            ValueError, match="Origins referenced in lanes but missing from origins_df"
        ):
            optimizer.solve(
                valid_demand_ts,
                origins_df,
                lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )


# ---------------------------------------------------------------------------
# Tests: Variable count overflow raises ValueError
# ---------------------------------------------------------------------------


class TestVariableCountOverflow:
    """Test variable count overflow raises ValueError."""

    def test_variable_count_exceeds_limit(self):
        """Create a scenario where flow_vars + inventory_vars > 1,000,000."""
        # We need n_lanes * n_periods + n_destinations * n_periods > 1,000,000
        # Use 500 lanes and 2001 periods: 500*2001 + 1*2001 = 1,002,501 > 1,000,000
        n_lanes = 500
        n_periods = 2001

        demand_ts = pl.DataFrame(
            {
                "destination_id": ["D1"] * n_periods,
                "date": [date(2024, 1, 1)] * n_periods,  # dates don't matter for count
                "demand": [1.0] * n_periods,
            }
        )
        origins_df = pl.DataFrame(
            {
                "origin_id": [f"O{i}" for i in range(n_lanes)],
                "daily_capacity": [100.0] * n_lanes,
            }
        )
        lanes_df = pl.DataFrame(
            {
                "origin_id": [f"O{i}" for i in range(n_lanes)],
                "destination_id": ["D1"] * n_lanes,
                "unit_cost": [1.0] * n_lanes,
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})
        planning_horizon = [
            date(2024, 1, 1) + __import__("datetime").timedelta(days=i)
            for i in range(n_periods)
        ]

        optimizer = MultiPeriodOptimizer()
        with pytest.raises(ValueError, match="Variable count .* exceeds maximum limit"):
            optimizer.solve(
                demand_ts,
                origins_df,
                lanes_df,
                destinations_df,
                planning_horizon,
            )


# ---------------------------------------------------------------------------
# Tests: Invalid solver name raises ValueError
# ---------------------------------------------------------------------------


class TestSolverValidation:
    """Test invalid solver name raises ValueError."""

    def test_invalid_solver_name_raises(self):
        with pytest.raises(ValueError, match="Unsupported solver 'INVALID'"):
            MultiPeriodOptimizer(solver_name="INVALID")

    def test_invalid_solver_includes_supported_list(self):
        with pytest.raises(ValueError, match="Supported solvers:"):
            MultiPeriodOptimizer(solver_name="CPLEX")

    def test_valid_solver_glop(self):
        optimizer = MultiPeriodOptimizer(solver_name="GLOP")
        assert optimizer._solver_name == "GLOP"

    def test_valid_solver_cbc(self):
        optimizer = MultiPeriodOptimizer(solver_name="CBC")
        assert optimizer._solver_name == "CBC"

    def test_default_solver_is_glop(self):
        optimizer = MultiPeriodOptimizer()
        assert optimizer._solver_name == "GLOP"


# ---------------------------------------------------------------------------
# Tests: max_variables parameter
# ---------------------------------------------------------------------------


class TestMaxVariablesParameter:
    """Test the configurable max_variables parameter on MultiPeriodOptimizer."""

    def test_default_max_variables(self):
        """Default max_variables matches the module-level MAX_VARIABLES constant."""
        from optimization.validation import MAX_VARIABLES

        optimizer = MultiPeriodOptimizer()
        assert optimizer._max_variables == MAX_VARIABLES

    def test_custom_max_variables_accepted(self):
        """A custom positive max_variables is stored correctly."""
        optimizer = MultiPeriodOptimizer(max_variables=500_000)
        assert optimizer._max_variables == 500_000

    def test_zero_max_variables_raises(self):
        """max_variables=0 raises ValueError."""
        with pytest.raises(
            ValueError, match="max_variables must be a positive integer"
        ):
            MultiPeriodOptimizer(max_variables=0)

    def test_negative_max_variables_raises(self):
        """Negative max_variables raises ValueError."""
        with pytest.raises(
            ValueError, match="max_variables must be a positive integer"
        ):
            MultiPeriodOptimizer(max_variables=-1)

    def test_custom_limit_enforced_on_solve(
        self,
        valid_demand_ts,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        """Setting max_variables=1 causes solve() to raise ValueError for any real problem."""
        optimizer = MultiPeriodOptimizer(max_variables=1)
        with pytest.raises(ValueError, match="Variable count .* exceeds maximum limit"):
            optimizer.solve(
                valid_demand_ts,
                valid_origins_df,
                valid_lanes_df,
                valid_destinations_df,
                valid_planning_horizon,
            )


# ---------------------------------------------------------------------------
# Tests: End-to-end integration
# ---------------------------------------------------------------------------


class TestIntegrationEndToEnd:
    """Integration tests verifying end-to-end multi-period solve with known-optimal problems."""

    def test_known_optimal_2_origins_2_destinations_3_periods(self):
        """Verify flows, inventory, and total_cost match hand-computed expected values.

        Problem setup:
        - 2 origins: O1 (capacity=30), O2 (capacity=20)
        - 2 destinations: D1, D2
        - 3 periods: 2024-01-01, 2024-01-02, 2024-01-03
        - Lanes: O1→D1 (cost=1), O1→D2 (cost=3), O2→D1 (cost=4), O2→D2 (cost=2)
        - Demand: D1 gets 20 each period, D2 gets 15 each period
        - No holding cost, no initial inventory

        Optimal cost analysis:
        - Cheapest way to serve D1: O1→D1 at cost 1 (vs O2→D1 at cost 4)
        - Cheapest way to serve D2: O2→D2 at cost 2 (vs O1→D2 at cost 3)
        - Total demand for D1 over 3 periods: 60, all served by O1 at cost 1 → 60
        - Total demand for D2 over 3 periods: 45, all served by O2 at cost 2 → 90
        - Total optimal cost: 150
        - Note: Without holding cost, the solver may distribute flows across
          periods differently (alternative optima), but total cost is unique.
        """
        planning_horizon = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

        demand_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"] * 3,
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 1, 3),
                ],
                "demand": [20.0, 15.0, 20.0, 15.0, 20.0, 15.0],
            }
        )

        origins_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O2"],
                "daily_capacity": [30.0, 20.0],
            }
        )

        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1", "O2", "O2"],
                "destination_id": ["D1", "D2", "D1", "D2"],
                "unit_cost": [1.0, 3.0, 4.0, 2.0],
            }
        )

        destinations_df = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
            }
        )

        optimizer = MultiPeriodOptimizer()
        result = optimizer.solve(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
        )

        # Verify total cost matches hand-computed optimal
        assert result.total_cost == pytest.approx(150.0, abs=1e-6)

        # Verify cost breakdown: no holding cost, so all cost is transportation
        assert result.transportation_cost == pytest.approx(150.0, abs=1e-4)
        assert result.holding_cost == pytest.approx(0.0, abs=1e-6)
        assert result.transportation_cost + result.holding_cost == pytest.approx(
            result.total_cost, abs=1e-4
        )

        # Verify flows DataFrame is non-empty and has correct schema
        assert not result.flows.is_empty()
        assert set(result.flows.columns) == {
            "origin_id",
            "destination_id",
            "period",
            "flow",
        }

        # Verify total flow to each destination across all periods equals total demand
        # D1 total demand = 60, D2 total demand = 45
        d1_total_flow = result.flows.filter(pl.col("destination_id") == "D1")[
            "flow"
        ].sum()
        d2_total_flow = result.flows.filter(pl.col("destination_id") == "D2")[
            "flow"
        ].sum()
        assert d1_total_flow == pytest.approx(60.0, abs=1e-6)
        assert d2_total_flow == pytest.approx(45.0, abs=1e-6)

        # Verify all D1 demand is served by O1 (cheapest route)
        d1_from_o1 = result.flows.filter(
            (pl.col("destination_id") == "D1") & (pl.col("origin_id") == "O1")
        )["flow"].sum()
        assert d1_from_o1 == pytest.approx(60.0, abs=1e-6)

        # Verify all D2 demand is served by O2 (cheapest route)
        d2_from_o2 = result.flows.filter(
            (pl.col("destination_id") == "D2") & (pl.col("origin_id") == "O2")
        )["flow"].sum()
        assert d2_from_o2 == pytest.approx(45.0, abs=1e-6)

        # Verify capacity constraints: no origin exceeds capacity in any period
        for period in planning_horizon:
            for origin_id, capacity in [("O1", 30.0), ("O2", 20.0)]:
                origin_period_flow = result.flows.filter(
                    (pl.col("origin_id") == origin_id) & (pl.col("period") == period)
                )["flow"].sum()
                assert origin_period_flow <= capacity + 1e-6

        # Verify inventory DataFrame: 2 destinations × 3 periods = 6 rows
        assert len(result.inventory) == 6

        # Verify inventory balance: for each destination, final inventory after
        # last period should be 0 (all demand met, no excess overall)
        for dest_id in ["D1", "D2"]:
            last_period_inv = result.inventory.filter(
                (pl.col("destination_id") == dest_id)
                & (pl.col("period") == date(2024, 1, 3))
            )["inventory"][0]
            assert last_period_inv == pytest.approx(0.0, abs=1e-6)

        # Verify all inventory values are non-negative
        assert (result.inventory["inventory"] >= -1e-6).all()


# ---------------------------------------------------------------------------
# Tests: Cost breakdown (transportation_cost + holding_cost)
# ---------------------------------------------------------------------------


class TestCostBreakdown:
    """Verify transportation_cost and holding_cost fields on MultiPeriodResult."""

    def test_no_holding_cost_column_breakdown(self):
        """When destinations_df has no holding_cost column, holding_cost == 0."""
        planning_horizon = [date(2024, 1, 1), date(2024, 1, 2)]
        demand_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D1"],
                "date": planning_horizon,
                "demand": [10.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [3.0]}
        )
        destinations_df = pl.DataFrame(
            {"destination_id": ["D1"]}
        )  # no holding_cost column

        result = MultiPeriodOptimizer().solve(
            demand_ts, origins_df, lanes_df, destinations_df, planning_horizon
        )

        assert result.holding_cost == pytest.approx(0.0, abs=1e-6)
        assert result.transportation_cost == pytest.approx(result.total_cost, abs=1e-4)
        # 20 units × cost 3 = 60
        assert result.transportation_cost == pytest.approx(60.0, abs=1e-4)

    def test_holding_cost_incurred_when_capacity_forces_early_shipment(self):
        """When capacity forces over-shipment in period 1, holding cost is non-zero.

        Setup: 1 origin (capacity=10), 1 destination, 2 periods.
        Demand: period 1 = 5, period 2 = 15. Total = 20 = total capacity (10 × 2).
        The only feasible solution is ship 10 in each period, leaving 5 units in
        inventory after period 1.
          transport_cost = (10 + 10) × unit_cost(2) = 40
          holding_cost   = 5 × holding_cost(1)      = 5
          total_cost     = 45
        """
        planning_horizon = [date(2024, 1, 1), date(2024, 1, 2)]
        demand_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D1"],
                "date": planning_horizon,
                "demand": [5.0, 15.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [10.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [2.0]}
        )
        destinations_df = pl.DataFrame(
            {"destination_id": ["D1"], "holding_cost": [1.0]}
        )

        result = MultiPeriodOptimizer().solve(
            demand_ts, origins_df, lanes_df, destinations_df, planning_horizon
        )

        assert result.total_cost == pytest.approx(45.0, abs=1e-4)
        assert result.transportation_cost == pytest.approx(40.0, abs=1e-4)
        assert result.holding_cost == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: Multi-period lead time validation and behavior
# ---------------------------------------------------------------------------


class TestLeadTimeValidation:
    """Verify validation for negative lead_time_days."""

    def test_negative_lead_time_raises_value_error(self):
        planning_horizon = [date(2024, 1, 1), date(2024, 1, 2)]
        demand_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D1"],
                "date": planning_horizon,
                "demand": [10.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1"],
                "destination_id": ["D1"],
                "unit_cost": [1.0],
                "lead_time_days": [-1],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})

        with pytest.raises(ValueError, match="Negative lead_time_days values found"):
            MultiPeriodOptimizer().solve(
                demand_ts, origins_df, lanes_df, destinations_df, planning_horizon
            )


class TestLeadTimeBehavior:
    """Verify LP behavior with 1-day or multi-day lead times."""

    def test_1day_lead_time_flow_dispatched_day_before(self):
        # Demand on day 1 = 0, day 2 = 10
        # Lane lead time = 1 day
        # Inflow on day 2 arrives from flow dispatched on day 1.
        planning_horizon = [date(2024, 1, 1), date(2024, 1, 2)]
        demand_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D1"],
                "date": planning_horizon,
                "demand": [0.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1"],
                "destination_id": ["D1"],
                "unit_cost": [2.0],
                "lead_time_days": [1],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})

        result = MultiPeriodOptimizer().solve(
            demand_ts, origins_df, lanes_df, destinations_df, planning_horizon
        )

        # Flow dispatched on day 1 (2024-01-01) must be 10.0 to arrive on day 2 (2024-01-02)
        day1_flow = result.flows.filter(
            (pl.col("origin_id") == "O1")
            & (pl.col("destination_id") == "D1")
            & (pl.col("period") == date(2024, 1, 1))
        )["flow"][0]
        assert day1_flow == pytest.approx(10.0, abs=1e-6)

        day2_flows = result.flows.filter(
            (pl.col("origin_id") == "O1")
            & (pl.col("destination_id") == "D1")
            & (pl.col("period") == date(2024, 1, 2))
        )
        assert len(day2_flows) == 0 or day2_flows["flow"][0] == pytest.approx(
            0.0, abs=1e-6
        )

        assert result.transportation_cost + result.holding_cost == pytest.approx(
            result.total_cost, abs=1e-4
        )

    def test_breakdown_sums_to_total_cost_with_holding(
        self,
        valid_demand_ts,
        valid_origins_df,
        valid_lanes_df,
        valid_destinations_df,
        valid_planning_horizon,
    ):
        """transportation_cost + holding_cost == total_cost for any valid solve."""
        result = MultiPeriodOptimizer().solve(
            valid_demand_ts,
            valid_origins_df,
            valid_lanes_df,
            valid_destinations_df,
            valid_planning_horizon,
        )

        assert result.transportation_cost >= 0.0
        assert result.holding_cost >= 0.0
        assert result.transportation_cost + result.holding_cost == pytest.approx(
            result.total_cost, abs=1e-4
        )
