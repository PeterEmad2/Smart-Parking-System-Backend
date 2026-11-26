from typing import Literal
from pydantic import BaseModel, Field

VehicleType = Literal["car", "motorcycle", "unknown"]


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


class ScanResponse(BaseModel):
    """Unified scan payload used by analyze and scan lookup endpoints."""

    scan_id: int
    filename: str | None = None
    plate_count: int
    plates: list[PlateDetection]
    annotated_image_url: str | None = None
    created_at: str


class HistoryResponse(BaseModel):
    """Paginated list of scan records."""

    total: int
    limit: int
    offset: int
    scans: list[ScanResponse]


class DeleteResponse(BaseModel):
    """Confirmation payload for DELETE /scans/{scan_id}."""

    deleted_scan_id: int


class HealthResponse(BaseModel):
    """Service health and inference readiness payload."""

    status: str
    device: str
    model_loaded: bool
