import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from SRC.utils.config import (
    APP_NAME,
    CHAR_CONF,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    PLATE_CONF,
    VERSION,
    get_device,
    is_model_loaded,
)
from SRC.utils.db import (
    checkout_parking_session,
    delete_parking_session,
    delete_scan,
    get_parking_rates,
    get_scan,
    init_db,
    list_parking_sessions,
    list_scans,
    process_parking_for_scan,
    save_scan,
    update_parking_rates,
)
from SRC.utils.inference import run_pipeline
from SRC.utils.response import (
    DeleteResponse,
    DeleteSessionResponse,
    GateMode,
    HealthResponse,
    HistoryResponse,
    ParkingRates,
    ParkingSessionRecord,
    ParkingSessionsResponse,
    ScanResponse,
    VehicleType,
)

logger = logging.getLogger(__name__)


class UpdateRatesPayload(BaseModel):
    car_rate_egp: float = Field(gt=0.0, le=5000.0)
    motorcycle_rate_egp: float = Field(gt=0.0, le=5000.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize SQLite storage and Smart Parking tables on startup."""
    init_db()
    yield


app = FastAPI(
    title=APP_NAME,
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Scan-Id", "X-Plate-Count", "X-Annotated-Image-Url"],
)


def _to_scan_response(scan: dict, request: Request) -> ScanResponse:
    """Map a database scan dict into a public ScanResponse with its image URL and parking events."""
    image_url = (
        str(request.url_for("get_scan_image", scan_id=scan["scan_id"]))
        if scan.get("has_annotated_image")
        else None
    )
    return ScanResponse(
        scan_id=scan["scan_id"],
        filename=scan.get("filename"),
        plate_count=scan["plate_count"],
        plates=scan["plates"],
        annotated_image_url=image_url,
        created_at=scan["created_at"],
        parking_events=scan.get("parking_events", []),
    )


@app.get("/")
def root():
    return {"name": APP_NAME, "version": VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=str(get_device()),
        model_loaded=is_model_loaded(),
    )


@app.post(
    "/analyze",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Annotated JPEG image when format=image",
        },
        201: {
            "model": ScanResponse,
            "description": "Scan metadata, annotated_image_url, and smart parking entry/exit events",
        },
    },
)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    format: Literal["json", "image"] = Query("json"),
    plate_conf: float = Query(PLATE_CONF, ge=0.0, le=1.0),
    char_conf: float = Query(CHAR_CONF, ge=0.0, le=1.0),
    gate_mode: GateMode = Query("auto"),
    simulate_minutes: int = Query(0, ge=0, le=10080),
    car_rate_egp: float | None = Query(None, gt=0.0, le=5000.0),
    motorcycle_rate_egp: float | None = Query(None, gt=0.0, le=5000.0),
):
    """Run plate detection + RTL Arabic OCR, match smart parking Entry/Exit, and persist records."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image uploaded")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_MB:g} MB",
            )

        result = run_pipeline(image_bytes, plate_conf=plate_conf, char_conf=char_conf)
        saved = save_scan(file.filename, result["plates"], result["annotated_image"])
        parking_events = process_parking_for_scan(
            scan_id=saved["scan_id"],
            plates=result["plates"],
            scan_time_iso=saved["created_at"],
            gate_mode=gate_mode,
            simulate_minutes=simulate_minutes,
            car_rate_override=car_rate_egp,
            moto_rate_override=motorcycle_rate_egp,
        )
        image_url = str(request.url_for("get_scan_image", scan_id=saved["scan_id"]))

        if format == "image":
            return Response(
                content=result["annotated_image"],
                media_type="image/jpeg",
                headers={
                    "X-Scan-Id": str(saved["scan_id"]),
                    "X-Plate-Count": str(len(result["plates"])),
                    "X-Annotated-Image-Url": image_url,
                },
            )

        payload = ScanResponse(
            scan_id=saved["scan_id"],
            filename=file.filename,
            plate_count=len(result["plates"]),
            plates=result["plates"],
            annotated_image_url=image_url,
            created_at=saved["created_at"],
            parking_events=parking_events,
        )
        return JSONResponse(content=payload.model_dump(), status_code=201)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during image analysis")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image analysis",
        )


@app.get("/parking/rates", response_model=ParkingRates)
def read_parking_rates():
    """Fetch current EGP hourly parking rates."""
    return ParkingRates(**get_parking_rates())


@app.put("/parking/rates", response_model=ParkingRates)
def write_parking_rates(payload: UpdateRatesPayload = Body(...)):
    """Update EGP hourly parking rates for cars and motorcycles."""
    updated = update_parking_rates(
        car_rate_egp=payload.car_rate_egp,
        motorcycle_rate_egp=payload.motorcycle_rate_egp,
    )
    return ParkingRates(**updated)


@app.get("/parking/sessions", response_model=ParkingSessionsResponse)
def get_parking_ledger(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Literal["all", "parked", "completed"] = Query("all"),
    plate: str | None = Query(None),
    vehicle_type: VehicleType | None = Query(None),
):
    """List smart parking sessions (active parked & completed exits) plus revenue KPIs."""
    status_filter = None if status == "all" else status
    total, sessions, summary = list_parking_sessions(
        limit=limit,
        offset=offset,
        status=status_filter,
        plate_query=plate,
        vehicle_type=vehicle_type,
    )
    return ParkingSessionsResponse(
        total=total,
        limit=limit,
        offset=offset,
        summary=summary,
        sessions=sessions,
    )


@app.post("/parking/sessions/{session_id}/checkout", response_model=ParkingSessionRecord)
def manual_session_checkout(
    session_id: int,
    simulate_minutes: int = Query(0, ge=0, le=10080),
):
    """Check out an active parked vehicle and calculate final duration & EGP fee."""
    checked_out = checkout_parking_session(session_id, simulate_minutes=simulate_minutes)
    if checked_out is None:
        raise HTTPException(status_code=404, detail=f"Parking session {session_id} not found")
    return ParkingSessionRecord(**checked_out)


@app.delete("/parking/sessions/{session_id}", response_model=DeleteSessionResponse)
def remove_parking_session(session_id: int):
    """Delete a parking session record."""
    if not delete_parking_session(session_id):
        raise HTTPException(status_code=404, detail=f"Parking session {session_id} not found")
    return DeleteSessionResponse(deleted_session_id=session_id)


@app.get("/scans", response_model=HistoryResponse)
def list_scan_history(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    plate: str | None = Query(None),
    vehicle_type: VehicleType | None = Query(None),
):
    """List persisted plate scans with optional Arabic plate text and vehicle type filtering."""
    total, scans = list_scans(
        limit=limit, offset=offset, plate_query=plate, vehicle_type=vehicle_type
    )
    return HistoryResponse(
        total=total,
        limit=limit,
        offset=offset,
        scans=[_to_scan_response(s, request) for s in scans],
    )


@app.get("/scans/{scan_id}", response_model=ScanResponse)
def get_scan_detail(scan_id: int, request: Request):
    """Fetch a single scan record by ID."""
    scan = get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    return _to_scan_response(scan, request)


@app.get("/scans/{scan_id}/image")
def get_scan_image(scan_id: int):
    """Stream the saved annotated JPEG for a scan record."""
    scan = get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")

    image_path = scan.get("annotated_image_path")
    if not image_path or not os.path.isfile(image_path):
        raise HTTPException(
            status_code=404, detail=f"Annotated image for scan {scan_id} not found"
        )

    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.delete("/scans/{scan_id}", response_model=DeleteResponse)
def delete_scan_record(scan_id: int):
    """Delete a scan record and its saved annotated image."""
    if not delete_scan(scan_id):
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    return DeleteResponse(deleted_scan_id=scan_id)
