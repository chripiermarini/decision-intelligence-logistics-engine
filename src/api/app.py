"""FastAPI application for the Decision Intelligence Logistics Engine."""

import logging

import polars as pl
from fastapi import FastAPI, HTTPException

from api.logistics_api import LogisticsAPI
from api.models import (
    FailedDestination,
    FlowRecord,
    ForecastedDestination,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    InventoryRecord,
    ModelResult,
    OptimizeRequest,
    OptimizeResponse,
    PlanRequest,
    PlanResponse,
)
from forecasting.config import PerDestinationConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Decision Intelligence Logistics Engine",
    description="Per-destination demand forecasting and multi-period flow optimisation.",
    version="1.1.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_forecast_config(
    model_names: list[str],
    train_ratio: float,
    selection_metric: str,
    max_workers: int,
    minimum_history_length: int,
    random_seed: int,
    model_params: dict,
) -> PerDestinationConfig:
    return PerDestinationConfig(
        model_names=model_names,
        train_ratio=train_ratio,
        selection_metric=selection_metric,
        max_workers=max_workers,
        minimum_history_length=minimum_history_length,
        random_seed=random_seed,
        model_params=model_params,
    )


def _to_demand_df(records) -> pl.DataFrame:
    return pl.DataFrame([r.model_dump() for r in records]).cast(
        {"date": pl.Date, "demand": pl.Float64}
    )


def _to_origins_df(records) -> pl.DataFrame:
    return pl.DataFrame([r.model_dump() for r in records])


def _to_lanes_df(records) -> pl.DataFrame:
    if any(getattr(r, "lead_time_days", 0) > 0 for r in records):
        return pl.DataFrame(
            [
                {
                    "origin_id": r.origin_id,
                    "destination_id": r.destination_id,
                    "unit_cost": r.unit_cost,
                    "lead_time_days": r.lead_time_days,
                }
                for r in records
            ]
        )
    return pl.DataFrame(
        [
            {
                "origin_id": r.origin_id,
                "destination_id": r.destination_id,
                "unit_cost": r.unit_cost,
            }
            for r in records
        ]
    )


def _to_destinations_df(records) -> pl.DataFrame:
    """Convert destination records to a DataFrame.

    ``holding_cost`` is included only when at least one destination provides a
    non-null value. Destinations that omit it default to ``0.0`` so the column
    stays numeric for the optimizer. When every destination omits holding cost,
    the column is left out entirely (shipping-cost-only objective).
    """
    if any(r.holding_cost is not None for r in records):
        return pl.DataFrame(
            [
                {
                    "destination_id": r.destination_id,
                    "holding_cost": (
                        0.0 if r.holding_cost is None else float(r.holding_cost)
                    ),
                }
                for r in records
            ]
        )
    return pl.DataFrame([{"destination_id": r.destination_id} for r in records])


def _build_forecast_response(result) -> ForecastResponse:
    """Convert AggregatedForecastingResult → ForecastResponse."""
    successful = []
    for outcome in result.successful:
        selected_name = outcome.selected.model_name
        selected_fr = next(
            (fr for fr in outcome.results if fr.model_name == selected_name), None
        )
        forecast_values = []
        if selected_fr is not None:
            forecast_values = [
                ForecastPoint(date=row["date"], forecast=row["forecast"])
                for row in selected_fr.forecast_values.iter_rows(named=True)
            ]
        successful.append(
            ForecastedDestination(
                destination_id=outcome.destination_id,
                best_model=selected_name,
                metrics=outcome.selected.metrics,
                forecast_values=forecast_values,
                all_models=[
                    ModelResult(model_name=fr.model_name, metrics=fr.metrics)
                    for fr in outcome.results
                ],
            )
        )

    failed = [
        FailedDestination(destination_id=o.destination_id, error=o.error)
        for o in result.failed
    ]

    return ForecastResponse(
        successful=successful,
        failed=failed,
        n_successful=len(successful),
        n_failed=len(failed),
    )


def _build_optimize_response(result) -> OptimizeResponse:
    """Convert MultiPeriodResult → OptimizeResponse."""
    flows = [
        FlowRecord(
            origin_id=row["origin_id"],
            destination_id=row["destination_id"],
            period=row["period"],
            flow=row["flow"],
        )
        for row in result.flows.iter_rows(named=True)
    ]
    inventory = [
        InventoryRecord(
            destination_id=row["destination_id"],
            period=row["period"],
            inventory=row["inventory"],
        )
        for row in result.inventory.iter_rows(named=True)
    ]
    return OptimizeResponse(
        total_cost=result.total_cost,
        transportation_cost=result.transportation_cost,
        holding_cost=result.holding_cost,
        flows=flows,
        inventory=inventory,
    )


def _extract_demand_ts(forecast_result) -> pl.DataFrame:
    """Extract [destination_id, date, demand] from the selected model forecasts."""
    frames = []
    for outcome in forecast_result.successful:
        selected_name = outcome.selected.model_name
        selected_fr = next(
            (fr for fr in outcome.results if fr.model_name == selected_name), None
        )
        if selected_fr is None:
            continue
        dest_df = (
            selected_fr.forecast_values.drop_nulls("forecast")
            .with_columns(pl.lit(outcome.destination_id).alias("destination_id"))
            .rename({"forecast": "demand"})
            .select(["destination_id", "date", "demand"])
        )
        frames.append(dest_df)

    if not frames:
        raise ValueError("No forecast data available to optimise over.")

    return pl.concat(frames).sort(["destination_id", "date"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Run per-destination forecasting on historical demand data.

    Trains and evaluates the configured models independently for each
    destination. Returns the best model per destination with its
    test-period forecast values and accuracy metrics.
    """
    if not request.demand_history:
        raise HTTPException(status_code=422, detail="demand_history must not be empty.")

    try:
        config = _build_forecast_config(
            model_names=request.model_names,
            train_ratio=request.train_ratio,
            selection_metric=request.selection_metric,
            max_workers=request.max_workers,
            minimum_history_length=request.minimum_history_length,
            random_seed=request.random_seed,
            model_params=request.model_params,
        )
        demand_df = _to_demand_df(request.demand_history)
        api = LogisticsAPI(config=config)
        result = api.forecast(demand_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during /forecast")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_forecast_response(result)


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Solve the multi-period minimum-cost transportation problem.

    Takes a demand time series (e.g. from /forecast) and the logistics
    network (origins, lanes, destinations) and returns the optimal
    origin-to-destination flow allocation for each period, together
    with per-destination inventory levels and total cost.
    """
    if not request.demand_ts:
        raise HTTPException(status_code=422, detail="demand_ts must not be empty.")
    if not request.origins:
        raise HTTPException(status_code=422, detail="origins must not be empty.")
    if not request.lanes:
        raise HTTPException(status_code=422, detail="lanes must not be empty.")

    try:
        demand_ts = _to_demand_df(request.demand_ts)
        origins_df = _to_origins_df(request.origins)
        lanes_df = _to_lanes_df(request.lanes)
        destinations_df = _to_destinations_df(request.destinations)
        planning_horizon = sorted(demand_ts["date"].unique().to_list())

        config = PerDestinationConfig(model_names=["naive_forecaster"])
        api = LogisticsAPI(config=config)
        result = api.optimize(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
            initial_inventory=request.initial_inventory or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during /optimize")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_optimize_response(result)


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest) -> PlanResponse:
    """Run the full pipeline: forecast → optimize in a single call.

    Combines /forecast and /optimize: trains per-destination models,
    extracts the selected forecasts as a demand time series (dropping
    null-valued leading points), and feeds them directly into the
    multi-period optimizer.
    """
    if not request.demand_history:
        raise HTTPException(status_code=422, detail="demand_history must not be empty.")

    try:
        config = _build_forecast_config(
            model_names=request.model_names,
            train_ratio=request.train_ratio,
            selection_metric=request.selection_metric,
            max_workers=request.max_workers,
            minimum_history_length=request.minimum_history_length,
            random_seed=request.random_seed,
            model_params=request.model_params,
        )
        demand_df = _to_demand_df(request.demand_history)
        origins_df = _to_origins_df(request.origins)
        lanes_df = _to_lanes_df(request.lanes)
        destinations_df = _to_destinations_df(request.destinations)

        api = LogisticsAPI(config=config)

        # Step 1: forecast
        forecast_result = api.forecast(demand_df)

        # Step 2: extract demand time series from selected forecasts
        demand_ts = _extract_demand_ts(forecast_result)
        planning_horizon = sorted(demand_ts["date"].unique().to_list())

        # Step 3: optimize
        opt_result = api.optimize(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
            initial_inventory=request.initial_inventory or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during /plan")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PlanResponse(
        forecast=_build_forecast_response(forecast_result),
        optimization=_build_optimize_response(opt_result),
    )
