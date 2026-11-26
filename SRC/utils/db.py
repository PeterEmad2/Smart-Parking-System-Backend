import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .config import DB_PATH, OUTPUT_DIR

DEFAULT_CAR_RATE_EGP = 25.0
DEFAULT_MOTO_RATE_EGP = 10.0


def normalize_plate_text(text: str) -> str:
    """Normalize Arabic plate text so spaces and Alef/Haa glyph variants match reliably."""
    cleaned = "".join(ch for ch in (text or "") if not ch.isspace())
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ھ": "ه",
        "ة": "ه",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned.strip()


def calculate_parking_fee(duration_minutes: int, hourly_rate_egp: float) -> tuple[float, float]:
    """Calculate billable hours (1h minimum, then 30m increments) and total fee in EGP."""
    clamped_mins = max(0, int(duration_minutes))
    if clamped_mins <= 60:
        billable_hours = 1.0
    else:
        half_hour_blocks = math.ceil(clamped_mins / 30.0)
        billable_hours = half_hour_blocks * 0.5
    total_fee = round(billable_hours * float(hourly_rate_egp), 2)
    return billable_hours, total_fee


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@contextmanager
def _get_connection():
    """Open a SQLite connection with foreign keys and WAL enabled."""
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create database tables/indexes and run additive schema migrations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with _get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                plate_count INTEGER NOT NULL,
                annotated_image_path TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
                plate_index INTEGER NOT NULL,
                vehicle_type TEXT NOT NULL DEFAULT 'unknown',
                precision REAL NOT NULL,
                text TEXT NOT NULL,
                characters_json TEXT NOT NULL,
                bbox_json TEXT NOT NULL,
                fallback_full_image INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS parking_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                car_rate_egp REAL NOT NULL DEFAULT 25.0,
                motorcycle_rate_egp REAL NOT NULL DEFAULT 10.0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS parking_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                plate_normalized TEXT NOT NULL,
                vehicle_type TEXT NOT NULL DEFAULT 'car',
                status TEXT NOT NULL DEFAULT 'parked',
                entry_scan_id INTEGER REFERENCES scans(id) ON DELETE SET NULL,
                exit_scan_id INTEGER REFERENCES scans(id) ON DELETE SET NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                duration_minutes INTEGER NOT NULL DEFAULT 0,
                billable_hours REAL NOT NULL DEFAULT 1.0,
                hourly_rate_egp REAL NOT NULL DEFAULT 25.0,
                total_fee_egp REAL NOT NULL DEFAULT 25.0
            );
            """
        )

        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(plates)").fetchall()
        }
        if "vehicle_type" not in existing_cols:
            conn.execute(
                "ALTER TABLE plates ADD COLUMN vehicle_type TEXT NOT NULL DEFAULT 'unknown'"
            )

        conn.execute(
            """
            INSERT OR IGNORE INTO parking_settings (id, car_rate_egp, motorcycle_rate_egp, updated_at)
            VALUES (1, ?, ?, ?)
            """,
            (DEFAULT_CAR_RATE_EGP, DEFAULT_MOTO_RATE_EGP, datetime.now(timezone.utc).isoformat()),
        )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_plates_scan_id ON plates(scan_id);
            CREATE INDEX IF NOT EXISTS idx_plates_text ON plates(text);
            CREATE INDEX IF NOT EXISTS idx_plates_vehicle_type ON plates(vehicle_type);
            CREATE INDEX IF NOT EXISTS idx_parking_norm_status ON parking_sessions(plate_normalized, status);
            CREATE INDEX IF NOT EXISTS idx_parking_entry_scan ON parking_sessions(entry_scan_id);
            CREATE INDEX IF NOT EXISTS idx_parking_exit_scan ON parking_sessions(exit_scan_id);
            """
        )


def get_parking_rates() -> dict:
    """Return current car and motorcycle hourly rates in EGP."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT car_rate_egp, motorcycle_rate_egp FROM parking_settings WHERE id = 1"
        ).fetchone()
    car_rate = float(row["car_rate_egp"]) if row else DEFAULT_CAR_RATE_EGP
    moto_rate = float(row["motorcycle_rate_egp"]) if row else DEFAULT_MOTO_RATE_EGP
    return {
        "car_rate_egp": car_rate,
        "motorcycle_rate_egp": moto_rate,
        "min_hours": 1.0,
        "rounding_minutes": 30,
        "currency": "EGP",
    }


def update_parking_rates(car_rate_egp: float, motorcycle_rate_egp: float) -> dict:
    """Update persisted EGP hourly parking rates."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO parking_settings (id, car_rate_egp, motorcycle_rate_egp, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                car_rate_egp = excluded.car_rate_egp,
                motorcycle_rate_egp = excluded.motorcycle_rate_egp,
                updated_at = excluded.updated_at
            """,
            (float(car_rate_egp), float(motorcycle_rate_egp), now_iso),
        )
    return get_parking_rates()


def _row_to_session(row: sqlite3.Row, event_override: str | None = None) -> dict:
    """Convert a parking_sessions row to a serializable dict with live elapsed estimates if parked."""
    status = row["status"]
    hourly_rate = float(row["hourly_rate_egp"])
    entry_time = row["entry_time"]
    exit_time = row["exit_time"]

    if status == "parked" and not exit_time:
        elapsed_secs = (datetime.now(timezone.utc) - _parse_iso(entry_time)).total_seconds()
        duration_mins = max(0, int(elapsed_secs // 60))
        billable_hours, total_fee = calculate_parking_fee(duration_mins, hourly_rate)
        event_type = event_override or "entry"
    else:
        duration_mins = int(row["duration_minutes"] or 0)
        billable_hours = float(row["billable_hours"] or 1.0)
        total_fee = float(row["total_fee_egp"] or hourly_rate)
        event_type = event_override or "exit"

    return {
        "session_id": row["id"],
        "event_type": event_type,
        "status": status,
        "plate_text": row["plate_text"],
        "plate_normalized": row["plate_normalized"],
        "vehicle_type": row["vehicle_type"] or "car",
        "entry_scan_id": row["entry_scan_id"],
        "exit_scan_id": row["exit_scan_id"],
        "entry_time": entry_time,
        "exit_time": exit_time,
        "duration_minutes": duration_mins,
        "billable_hours": billable_hours,
        "hourly_rate_egp": hourly_rate,
        "total_fee_egp": total_fee,
    }


def process_parking_for_scan(
    scan_id: int,
    plates: list[dict],
    scan_time_iso: str,
    gate_mode: str = "auto",
    simulate_minutes: int = 0,
    car_rate_override: float | None = None,
    moto_rate_override: float | None = None,
) -> list[dict]:
    """Match detected plates against open parking sessions to open Entry or calculate Exit fee."""
    rates = get_parking_rates()
    car_rate = float(car_rate_override) if car_rate_override is not None else rates["car_rate_egp"]
    moto_rate = (
        float(moto_rate_override)
        if moto_rate_override is not None
        else rates["motorcycle_rate_egp"]
    )

    events: list[dict] = []
    seen_norm: set[str] = set()

    with _get_connection() as conn:
        for p in plates:
            raw_text = (p.get("text") or "").strip()
            norm = normalize_plate_text(raw_text)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)

            v_type = p.get("vehicle_type") or "car"
            if v_type not in ("car", "motorcycle"):
                v_type = "car"
            rate = moto_rate if v_type == "motorcycle" else car_rate

            active_row = conn.execute(
                """
                SELECT * FROM parking_sessions
                WHERE plate_normalized = ? AND status = 'parked'
                ORDER BY id DESC
                LIMIT 1
                """,
                (norm,),
            ).fetchone()

            should_checkout = (gate_mode == "auto" and active_row is not None) or (
                gate_mode == "exit"
            )

            if should_checkout and active_row is not None:
                entry_dt = _parse_iso(active_row["entry_time"])
                now_dt = _parse_iso(scan_time_iso) + timedelta(minutes=max(0, int(simulate_minutes)))
                if now_dt < entry_dt:
                    now_dt = entry_dt
                duration_mins = max(0, int((now_dt - entry_dt).total_seconds() // 60))
                applied_rate = float(active_row["hourly_rate_egp"] or rate)
                billable_hours, total_fee = calculate_parking_fee(duration_mins, applied_rate)
                exit_iso = now_dt.isoformat()

                conn.execute(
                    """
                    UPDATE parking_sessions
                    SET status = 'completed',
                        exit_scan_id = ?,
                        exit_time = ?,
                        duration_minutes = ?,
                        billable_hours = ?,
                        total_fee_egp = ?
                    WHERE id = ?
                    """,
                    (scan_id, exit_iso, duration_mins, billable_hours, total_fee, active_row["id"]),
                )
                updated = conn.execute(
                    "SELECT * FROM parking_sessions WHERE id = ?", (active_row["id"],)
                ).fetchone()
                events.append(_row_to_session(updated, event_override="exit"))
            elif gate_mode == "exit" and active_row is None:
                # Direct exit override without a prior open session: create an instant completed receipt
                duration_mins = max(60, int(simulate_minutes or 60))
                exit_dt = _parse_iso(scan_time_iso)
                entry_dt = exit_dt - timedelta(minutes=duration_mins)
                billable_hours, total_fee = calculate_parking_fee(duration_mins, rate)
                cur = conn.execute(
                    """
                    INSERT INTO parking_sessions (
                        plate_text, plate_normalized, vehicle_type, status,
                        entry_scan_id, exit_scan_id, entry_time, exit_time,
                        duration_minutes, billable_hours, hourly_rate_egp, total_fee_egp
                    )
                    VALUES (?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw_text,
                        norm,
                        v_type,
                        scan_id,
                        scan_id,
                        entry_dt.isoformat(),
                        exit_dt.isoformat(),
                        duration_mins,
                        billable_hours,
                        rate,
                        total_fee,
                    ),
                )
                created = conn.execute(
                    "SELECT * FROM parking_sessions WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
                events.append(_row_to_session(created, event_override="exit"))
            else:
                # Entry mode: if gate_mode == 'entry' and already parked, close old or refresh
                if active_row is not None and gate_mode == "entry":
                    events.append(_row_to_session(active_row, event_override="entry"))
                    continue

                billable_hours, min_fee = calculate_parking_fee(0, rate)
                cur = conn.execute(
                    """
                    INSERT INTO parking_sessions (
                        plate_text, plate_normalized, vehicle_type, status,
                        entry_scan_id, exit_scan_id, entry_time, exit_time,
                        duration_minutes, billable_hours, hourly_rate_egp, total_fee_egp
                    )
                    VALUES (?, ?, ?, 'parked', ?, NULL, ?, NULL, 0, ?, ?, ?)
                    """,
                    (
                        raw_text,
                        norm,
                        v_type,
                        scan_id,
                        scan_time_iso,
                        billable_hours,
                        rate,
                        min_fee,
                    ),
                )
                created = conn.execute(
                    "SELECT * FROM parking_sessions WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
                events.append(_row_to_session(created, event_override="entry"))

    return events


def checkout_parking_session(session_id: int, simulate_minutes: int = 0) -> dict | None:
    """Manually check out an active parking session (with optional simulated elapsed minutes)."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM parking_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        if row["status"] == "completed":
            return _row_to_session(row, event_override="exit")

        entry_dt = _parse_iso(row["entry_time"])
        exit_dt = datetime.now(timezone.utc) + timedelta(minutes=max(0, int(simulate_minutes)))
        if exit_dt < entry_dt:
            exit_dt = entry_dt
        duration_mins = max(0, int((exit_dt - entry_dt).total_seconds() // 60))
        rate = float(row["hourly_rate_egp"])
        billable_hours, total_fee = calculate_parking_fee(duration_mins, rate)

        conn.execute(
            """
            UPDATE parking_sessions
            SET status = 'completed',
                exit_time = ?,
                duration_minutes = ?,
                billable_hours = ?,
                total_fee_egp = ?
            WHERE id = ?
            """,
            (exit_dt.isoformat(), duration_mins, billable_hours, total_fee, session_id),
        )
        updated = conn.execute(
            "SELECT * FROM parking_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(updated, event_override="exit")


def delete_parking_session(session_id: int) -> bool:
    """Delete a parking session by ID."""
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM parking_sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


def get_parking_summary(conn: sqlite3.Connection | None = None) -> dict:
    """Compute live parking occupancy and completed EGP revenue summary."""
    rates = get_parking_rates()

    def _query(c: sqlite3.Connection) -> dict:
        active_rows = c.execute(
            "SELECT vehicle_type, COUNT(*) AS cnt FROM parking_sessions WHERE status = 'parked' GROUP BY vehicle_type"
        ).fetchall()
        active_cars = 0
        active_motos = 0
        total_active = 0
        for r in active_rows:
            cnt = int(r["cnt"])
            total_active += cnt
            if r["vehicle_type"] == "motorcycle":
                active_motos += cnt
            else:
                active_cars += cnt

        comp = c.execute(
            """
            SELECT COUNT(*) AS cnt, COALESCE(SUM(total_fee_egp), 0.0) AS rev
            FROM parking_sessions
            WHERE status = 'completed'
            """
        ).fetchone()
        return {
            "active_parked_count": total_active,
            "active_cars": active_cars,
            "active_motorcycles": active_motos,
            "completed_count": int(comp["cnt"] or 0),
            "total_revenue_egp": round(float(comp["rev"] or 0.0), 2),
            "rates": rates,
        }

    if conn is not None:
        return _query(conn)
    with _get_connection() as managed:
        return _query(managed)


def list_parking_sessions(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    plate_query: str | None = None,
    vehicle_type: str | None = None,
) -> tuple[int, list[dict], dict]:
    """List parking sessions with optional status, plate text, and vehicle type filters."""
    conditions: list[str] = []
    params: list = []

    if status in ("parked", "completed"):
        conditions.append("status = ?")
        params.append(status)

    if plate_query and plate_query.strip():
        q = plate_query.strip()
        q_norm = normalize_plate_text(q)
        conditions.append("(plate_text LIKE ? OR plate_normalized LIKE ?)")
        params.extend([f"%{q}%", f"%{q_norm}%"])

    if vehicle_type in ("car", "motorcycle"):
        conditions.append("vehicle_type = ?")
        params.append(vehicle_type)

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with _get_connection() as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM parking_sessions {where_sql}", tuple(params)
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT * FROM parking_sessions
            {where_sql}
            ORDER BY CASE WHEN status = 'parked' THEN 0 ELSE 1 END, id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        sessions = [_row_to_session(r) for r in rows]
        summary = get_parking_summary(conn)
        return total, sessions, summary


def _fetch_parking_events_for_scans(
    conn: sqlite3.Connection, scan_ids: list[int]
) -> dict[int, list[dict]]:
    """Attach linked entry/exit parking session records to each scan_id."""
    if not scan_ids:
        return {}
    placeholders = ",".join("?" for _ in scan_ids)
    rows = conn.execute(
        f"""
        SELECT * FROM parking_sessions
        WHERE entry_scan_id IN ({placeholders}) OR exit_scan_id IN ({placeholders})
        ORDER BY id DESC
        """,
        (*scan_ids, *scan_ids),
    ).fetchall()

    grouped: dict[int, list[dict]] = {sid: [] for sid in scan_ids}
    for row in rows:
        entry_sid = row["entry_scan_id"]
        exit_sid = row["exit_scan_id"]
        if exit_sid in grouped and row["status"] == "completed":
            grouped[exit_sid].append(_row_to_session(row, event_override="exit"))
        if entry_sid in grouped and entry_sid != exit_sid:
            grouped[entry_sid].append(_row_to_session(row, event_override="entry"))
    return grouped


def _row_to_plate(row: sqlite3.Row) -> dict:
    """Convert a database plate row to an API-ready dict."""
    characters = json.loads(row["characters_json"])
    return {
        "plate_index": row["plate_index"],
        "vehicle_type": row["vehicle_type"] or "unknown",
        "precision": row["precision"],
        "text": row["text"],
        "characters": characters,
        "bbox": json.loads(row["bbox_json"]),
        "fallback_full_image": bool(row["fallback_full_image"]),
        "is_partial": len(characters) < 3,
    }


def _fetch_plates_for_scans(
    conn: sqlite3.Connection, scan_ids: list[int]
) -> dict[int, list[dict]]:
    """Load plate rows grouped by scan_id."""
    if not scan_ids:
        return {}
    placeholders = ",".join("?" for _ in scan_ids)
    rows = conn.execute(
        f"""
        SELECT scan_id, plate_index, vehicle_type, precision, text,
               characters_json, bbox_json, fallback_full_image
        FROM plates
        WHERE scan_id IN ({placeholders})
        ORDER BY scan_id DESC, plate_index ASC
        """,
        scan_ids,
    ).fetchall()
    grouped: dict[int, list[dict]] = {sid: [] for sid in scan_ids}
    for row in rows:
        grouped[row["scan_id"]].append(_row_to_plate(row))
    return grouped


def _build_scan_dict(
    scan_row: sqlite3.Row,
    plates: list[dict],
    parking_events: list[dict] | None = None,
) -> dict:
    """Build a scan dict including internal annotated_image_path and parking_events."""
    img_path = scan_row["annotated_image_path"]
    return {
        "scan_id": scan_row["id"],
        "filename": scan_row["filename"],
        "plate_count": scan_row["plate_count"],
        "plates": plates,
        "has_annotated_image": bool(img_path and os.path.isfile(img_path)),
        "annotated_image_path": img_path,
        "created_at": scan_row["created_at"],
        "parking_events": parking_events or [],
    }


def save_scan(filename: str | None, plates: list[dict], annotated_bytes: bytes) -> dict:
    """Persist an annotated JPEG and scan/plate records in SQLite."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.jpg")
    with open(image_path, "wb") as f:
        f.write(annotated_bytes)

    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scans (filename, plate_count, annotated_image_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (filename, len(plates), image_path, created_at),
            )
            scan_id = cursor.lastrowid

            conn.executemany(
                """
                INSERT INTO plates (
                    scan_id, plate_index, vehicle_type, precision, text,
                    characters_json, bbox_json, fallback_full_image, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        scan_id,
                        p["plate_index"],
                        p.get("vehicle_type", "unknown"),
                        p["precision"],
                        p["text"],
                        json.dumps(p["characters"], ensure_ascii=False),
                        json.dumps(p["bbox"]),
                        1 if p.get("fallback_full_image") else 0,
                        created_at,
                    )
                    for p in plates
                ],
            )
    except Exception:
        if os.path.isfile(image_path):
            os.remove(image_path)
        raise

    return {
        "scan_id": scan_id,
        "created_at": created_at,
        "annotated_image_path": image_path,
    }


def list_scans(
    limit: int = 20,
    offset: int = 0,
    plate_query: str | None = None,
    vehicle_type: str | None = None,
) -> tuple[int, list[dict]]:
    """Return paginated scan history, optionally filtered by plate text and/or vehicle type."""
    query = plate_query.strip() if plate_query else ""
    conditions = ["p.scan_id = s.id"]
    params: list = []

    if query:
        conditions.append("(p.text LIKE ? OR REPLACE(p.text, ' ', '') LIKE ?)")
        params.extend([f"%{query}%", f"%{query.replace(' ', '')}%"])

    if vehicle_type:
        conditions.append("p.vehicle_type = ?")
        params.append(vehicle_type)

    if len(conditions) > 1:
        where_clause = f"WHERE EXISTS (SELECT 1 FROM plates p WHERE {' AND '.join(conditions)})"
    else:
        where_clause = ""

    with _get_connection() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM scans s {where_clause}",
            tuple(params),
        ).fetchone()[0]

        scan_rows = conn.execute(
            f"""
            SELECT s.id, s.filename, s.plate_count, s.annotated_image_path, s.created_at
            FROM scans s
            {where_clause}
            ORDER BY s.id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

        scan_ids = [row["id"] for row in scan_rows]
        plates_by_scan = _fetch_plates_for_scans(conn, scan_ids)
        events_by_scan = _fetch_parking_events_for_scans(conn, scan_ids)
        scans = [
            _build_scan_dict(
                row,
                plates_by_scan.get(row["id"], []),
                events_by_scan.get(row["id"], []),
            )
            for row in scan_rows
        ]
        return total, scans


def get_scan(scan_id: int) -> dict | None:
    """Fetch a single scan, its plates, and linked parking events by scan_id."""
    with _get_connection() as conn:
        scan_row = conn.execute(
            """
            SELECT id, filename, plate_count, annotated_image_path, created_at
            FROM scans
            WHERE id = ?
            """,
            (scan_id,),
        ).fetchone()
        if scan_row is None:
            return None
        plates_by_scan = _fetch_plates_for_scans(conn, [scan_id])
        events_by_scan = _fetch_parking_events_for_scans(conn, [scan_id])
        return _build_scan_dict(
            scan_row,
            plates_by_scan.get(scan_id, []),
            events_by_scan.get(scan_id, []),
        )


def delete_scan(scan_id: int) -> bool:
    """Delete a scan, its cascade plates, and its stored annotated image."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT annotated_image_path FROM scans WHERE id = ?",
            (scan_id,),
        ).fetchone()
        if row is None:
            return False
        image_path = row["annotated_image_path"]
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))

    if image_path and os.path.isfile(image_path):
        os.remove(image_path)
    return True
