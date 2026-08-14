"""Pydantic request and response models for the logistics API."""

from datetime import date

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class DemandRecord(BaseModel):
    date: date
    destination_id: str
    demand: float


class OriginRecord(BaseModel):
    origin_id: str
    daily_capacity: float


class LaneRecord(BaseModel):
    origin_id: str
    destination_id: str
    unit_cost: float
    lead_time_days: int = Field(
        default=0,
        ge=0,
        description=(
            "Optional transit lead time in days. Defaults to 0 (instantaneous delivery)."
        ),
    )


class DestinationRecord(BaseModel):
    destination_id: str
    holding_cost: float | None = Field(
        default=None,
        description=(
            "Optional per-unit inventory holding cost. When provided for any "
            "destination, the optimizer includes holding costs in the objective."
        ),
    )


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------


class ForecastRequest(BaseModel):
    demand_history: list[DemandRecord]
    model_names: list[str] = Field(
        default=["naive_forecaster", "seasonal_forecaster", "rolling_window_forecaster"]
    )
    train_ratio: float = 0.8
    selection_metric: str = "wape"
    max_workers: int = 1
    minimum_history_length: int = 10
    random_seed: int = 42
    model_params: dict[str, dict] = Field(default_factory=dict)


class ForecastPoint(BaseModel):
    date: date
    forecast: float | None = None


class ModelResult(BaseModel):
    model_name: str
    metrics: dict[str, float]


class ForecastedDestination(BaseModel):
    destination_id: str
    best_model: str
    metrics: dict[str, float]
    forecast_values: list[ForecastPoint]
    all_models: list[ModelResult]


class FailedDestination(BaseModel):
    destination_id: str
    error: str | None = None


class ForecastResponse(BaseModel):
    successful: list[ForecastedDestination]
    failed: list[FailedDestination]
    n_successful: int
    n_failed: int


# ---------------------------------------------------------------------------
# /optimize
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    demand_ts: list[DemandRecord]
    origins: list[OriginRecord]
    lanes: list[LaneRecord]
    destinations: list[DestinationRecord]
    initial_inventory: dict[str, float] = Field(default_factory=dict)


class FlowRecord(BaseModel):
    origin_id: str
    destination_id: str
    period: date
    flow: float


class InventoryRecord(BaseModel):
    destination_id: str
    period: date
    inventory: float


class OptimizeResponse(BaseModel):
    total_cost: float
    transportation_cost: float
    holding_cost: float
    flows: list[FlowRecord]
    inventory: list[InventoryRecord]


# ---------------------------------------------------------------------------
# /plan  (forecast + optimize in one call)
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    demand_history: list[DemandRecord]
    origins: list[OriginRecord]
    lanes: list[LaneRecord]
    destinations: list[DestinationRecord]
    model_names: list[str] = Field(
        default=["naive_forecaster", "seasonal_forecaster", "rolling_window_forecaster"]
    )
    train_ratio: float = 0.8
    selection_metric: str = "wape"
    max_workers: int = 1
    minimum_history_length: int = 10
    random_seed: int = 42
    model_params: dict[str, dict] = Field(default_factory=dict)
    initial_inventory: dict[str, float] = Field(default_factory=dict)


class PlanResponse(BaseModel):
    forecast: ForecastResponse
    optimization: OptimizeResponse
