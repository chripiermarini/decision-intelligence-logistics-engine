"""Builds the multi-period transportation LP: lookups, variables, constraints, objective."""

from dataclasses import dataclass
from datetime import date

import polars as pl
from ortools.linear_solver import pywraplp

FlowVars = dict[tuple[str, str, date], pywraplp.Variable]
InventoryVars = dict[tuple[str, date], pywraplp.Variable]


@dataclass
class ModelLookups:
    """Pre-computed lookup structures shared by variable, constraint, and objective builders."""

    lanes_list: list[dict]
    capacity_map: dict[str, float]
    holding_cost_map: dict[str, float]
    has_holding_cost: bool
    demand_map: dict[tuple[str, date], float]
    destinations: list[str]
    origins: list[str]
    lanes_by_dest: dict[str, list[tuple[str, float]]]
    lanes_by_origin: dict[str, list[tuple[str, float]]]
    lead_time_map: dict[tuple[str, str], int]


def build_lookups(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame,
) -> ModelLookups:
    """Build the lookup structures used to construct variables, constraints, and the objective."""
    has_holding_cost = "holding_cost" in destinations_df.columns
    has_lead_time = "lead_time_days" in lanes_df.columns

    select_cols = ["origin_id", "destination_id", "unit_cost"]
    if has_lead_time:
        select_cols.append("lead_time_days")

    lanes_list = lanes_df.select(select_cols).to_dicts()

    lead_time_map: dict[tuple[str, str], int] = {}
    for lane in lanes_list:
        o_id, d_id = lane["origin_id"], lane["destination_id"]
        lt = int(lane.get("lead_time_days", 0) or 0)
        lead_time_map[(o_id, d_id)] = lt

    capacity_map: dict[str, float] = dict(
        zip(origins_df["origin_id"].to_list(), origins_df["daily_capacity"].to_list())
    )

    holding_cost_map: dict[str, float] = {}
    if has_holding_cost:
        holding_cost_map = dict(
            zip(
                destinations_df["destination_id"].to_list(),
                destinations_df["holding_cost"].to_list(),
            )
        )

    demand_map: dict[tuple[str, date], float] = {
        (row["destination_id"], row["date"]): row["demand"]
        for row in demand_ts.to_dicts()
    }

    destinations = sorted(demand_ts["destination_id"].unique().to_list())
    origins = sorted(origins_df["origin_id"].unique().to_list())

    lanes_by_dest: dict[str, list[tuple[str, float]]] = {}
    lanes_by_origin: dict[str, list[tuple[str, float]]] = {}
    for lane in lanes_list:
        o_id, d_id, cost = lane["origin_id"], lane["destination_id"], lane["unit_cost"]
        lanes_by_dest.setdefault(d_id, []).append((o_id, cost))
        lanes_by_origin.setdefault(o_id, []).append((d_id, cost))

    return ModelLookups(
        lanes_list=lanes_list,
        capacity_map=capacity_map,
        holding_cost_map=holding_cost_map,
        has_holding_cost=has_holding_cost,
        demand_map=demand_map,
        destinations=destinations,
        origins=origins,
        lanes_by_dest=lanes_by_dest,
        lanes_by_origin=lanes_by_origin,
        lead_time_map=lead_time_map,
    )


def create_variables(
    solver: pywraplp.Solver,
    lookups: ModelLookups,
    planning_horizon: list[date],
) -> tuple[FlowVars, InventoryVars]:
    """Create flow[o,d,t] >= 0 and inventory[d,t] >= 0 decision variables."""
    flow_vars: FlowVars = {}
    for lane in lookups.lanes_list:
        o_id, d_id = lane["origin_id"], lane["destination_id"]
        for t in planning_horizon:
            flow_vars[(o_id, d_id, t)] = solver.NumVar(
                0.0, solver.infinity(), f"flow_{o_id}_{d_id}_{t}"
            )

    inv_vars: InventoryVars = {}
    for d_id in lookups.destinations:
        for t in planning_horizon:
            inv_vars[(d_id, t)] = solver.NumVar(
                0.0, solver.infinity(), f"inv_{d_id}_{t}"
            )

    return flow_vars, inv_vars


def add_constraints(
    solver: pywraplp.Solver,
    lookups: ModelLookups,
    flow_vars: FlowVars,
    inv_vars: InventoryVars,
    planning_horizon: list[date],
    initial_inventory: dict[str, float],
) -> None:
    """Add inventory balance and origin capacity constraints.

    Inventory balance (per destination, per period):
        inv[d,0] = initial_inv[d] + inflow[d,0] - demand[d,0]
        inv[d,t] = inv[d,t-1] + inflow[d,t] - demand[d,t]   for t > 0
        where inflow[d,t] sums flow[o,d, t - L_od] for lanes with lead_time_days L_od.

    Capacity (per origin, per period):
        sum_d flow[o,d,t] <= daily_capacity[o]
    """
    for d_id in lookups.destinations:
        for t_idx, t in enumerate(planning_horizon):
            inflow_terms = []
            for o_id, _ in lookups.lanes_by_dest.get(d_id, []):
                lt = lookups.lead_time_map.get((o_id, d_id), 0)
                dispatch_idx = t_idx - lt
                if dispatch_idx >= 0:
                    dispatch_t = planning_horizon[dispatch_idx]
                    if (o_id, d_id, dispatch_t) in flow_vars:
                        inflow_terms.append(flow_vars[(o_id, d_id, dispatch_t)])

            demand_val = lookups.demand_map.get((d_id, t), 0.0)

            ct = solver.Constraint(0.0, 0.0, f"inv_bal_{d_id}_{t}")
            ct.SetCoefficient(inv_vars[(d_id, t)], 1.0)
            for flow_var in inflow_terms:
                ct.SetCoefficient(flow_var, -1.0)

            if t_idx == 0:
                prev_inv = initial_inventory.get(d_id, 0.0)
                ct.SetBounds(prev_inv - demand_val, prev_inv - demand_val)
            else:
                prev_t = planning_horizon[t_idx - 1]
                ct.SetCoefficient(inv_vars[(d_id, prev_t)], -1.0)
                ct.SetBounds(-demand_val, -demand_val)

    for o_id in lookups.origins:
        for t in planning_horizon:
            ct = solver.Constraint(0.0, lookups.capacity_map[o_id], f"cap_{o_id}_{t}")
            for d_id, _ in lookups.lanes_by_origin.get(o_id, []):
                if (o_id, d_id, t) in flow_vars:
                    ct.SetCoefficient(flow_vars[(o_id, d_id, t)], 1.0)


def set_objective(
    solver: pywraplp.Solver,
    lookups: ModelLookups,
    flow_vars: FlowVars,
    inv_vars: InventoryVars,
    planning_horizon: list[date],
) -> None:
    """Minimize total transportation cost plus (optional) holding cost.

    minimize sum unit_cost[o,d] * flow[o,d,t] + sum holding_cost[d] * inv[d,t]
    """
    objective = solver.Objective()
    objective.SetMinimization()

    for lane in lookups.lanes_list:
        o_id, d_id, cost = lane["origin_id"], lane["destination_id"], lane["unit_cost"]
        for t in planning_horizon:
            objective.SetCoefficient(flow_vars[(o_id, d_id, t)], cost)

    if lookups.has_holding_cost:
        for d_id in lookups.destinations:
            h_cost = lookups.holding_cost_map.get(d_id, 0.0)
            if h_cost > 0:
                for t in planning_horizon:
                    objective.SetCoefficient(inv_vars[(d_id, t)], h_cost)
