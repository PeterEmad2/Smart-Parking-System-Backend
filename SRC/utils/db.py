import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH, OUTPUT_DIR


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
            """
        )

        # Safe migration for existing databases created prior to vehicle_type support
        existing_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(plates)").fetchall()
        }
        if "vehicle_type" not in existing_cols:
            conn.execute(
                "ALTER TABLE plates ADD COLUMN vehicle_type TEXT NOT NULL DEFAULT 'unknown'"
            )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_plates_scan_id ON plates(scan_id);
            CREATE INDEX IF NOT EXISTS idx_plates_text ON plates(text);
            CREATE INDEX IF NOT EXISTS idx_plates_vehicle_type ON plates(vehicle_type);
            """
        )


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


def _fetch_plates_for_scans(conn: sqlite3.Connection, scan_ids: list[int]) -> dict[int, list[dict]]:
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


def _build_scan_dict(scan_row: sqlite3.Row, plates: list[dict]) -> dict:
    """Build a scan dict including internal annotated_image_path."""
    img_path = scan_row["annotated_image_path"]
    return {
        "scan_id": scan_row["id"],
        "filename": scan_row["filename"],
        "plate_count": scan_row["plate_count"],
        "plates": plates,
        "has_annotated_image": bool(img_path and os.path.isfile(img_path)),
        "annotated_image_path": img_path,
        "created_at": scan_row["created_at"],
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
        scans = [_build_scan_dict(row, plates_by_scan.get(row["id"], [])) for row in scan_rows]
        return total, scans


def get_scan(scan_id: int) -> dict | None:
    """Fetch a single scan and its plates by scan_id."""
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
        return _build_scan_dict(scan_row, plates_by_scan.get(scan_id, []))


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
