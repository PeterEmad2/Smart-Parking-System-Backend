from typing import Literal
from pydantic import BaseModel, Field

VehicleType = Literal["car", "motorcycle", "unknown"]
GateMode = Literal["auto", "entry", "exit"]
ParkingStatus = Literal["parked", "completed"]
ParkingEventType = Literal["entry", "exit"]


class PlateDetection(BaseModel):
    """Single detected license plate with vehicle type, precision, RTL characters, and bbox."""

    plate_index: int
    vehicle_type: VehicleType = "unknown"
    precision: float = Field(ge=0.0, le=1.0)
    text: str
    characters: list[str]
    bbox: list[int]
    fallback_full_image: bool = False
    is_partial: bool = False


class ParkingSessionRecord(BaseModel):
    """Smart parking session tracking entry, exit, duration, and EGP fee for a license plate."""

    session_id: int
    event_type: ParkingEventType = "entry"
    status: ParkingStatus
    plate_text: str
    plate_normalized: str
    vehicle_type: VehicleType = "car"
    entry_scan_id: int | None = None
    exit_scan_id: int | None = None
    entry_time: str
    exit_time: str | None = None
    duration_minutes: int = 0
    billable_hours: float = 1.0
    hourly_rate_egp: float
    total_fee_egp: float


class ParkingRates(BaseModel):
    """Editable EGP hourly parking tariff by vehicle class."""

    car_rate_egp: float = Field(default=25.0, gt=0.0, le=5000.0)
    motorcycle_rate_egp: float = Field(default=10.0, gt=0.0, le=5000.0)
    min_hours: float = 1.0
    rounding_minutes: int = 30
    currency: str = "EGP"


class ParkingSummary(BaseModel):
    """Aggregate smart parking occupancy and revenue metrics."""

    active_parked_count: int
    active_cars: int
    active_motorcycles: int
    completed_count: int
    total_revenue_egp: float
    rates: ParkingRates


class ParkingSessionsResponse(BaseModel):
    """Paginated smart parking sessions list with summary metrics."""

    total: int
    limit: int
    offset: int
    summary: ParkingSummary
    sessions: list[ParkingSessionRecord]


class ScanResponse(BaseModel):
    """Unified scan payload used by analyze and scan lookup endpoints."""

    scan_id: int
    filename: str | None = None
    plate_count: int
    plates: list[PlateDetection]
    annotated_image_url: str | None = None
    created_at: str
    parking_events: list[ParkingSessionRecord] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Paginated list of scan records."""

    total: int
    limit: int
    offset: int
    scans: list[ScanResponse]


class DeleteResponse(BaseModel):
    """Confirmation payload for DELETE /scans/{scan_id}."""

    deleted_scan_id: int


class DeleteSessionResponse(BaseModel):
    """Confirmation payload for DELETE /parking/sessions/{session_id}."""

    deleted_session_id: int


class HealthResponse(BaseModel):
    """Service health and inference readiness payload."""

    status: str
    device: str
    model_loaded: bool
