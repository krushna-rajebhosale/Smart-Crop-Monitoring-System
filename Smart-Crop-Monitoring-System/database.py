"""SQLite database for recommendations and scan history.

Tables
  recommendations : one row per model class (symptoms, treatment, prevention, nutrients,
                    plus 'source' and 'verified' so you can show where the text came from)
  scans           : one row per analysed photo (powers the dashboard)

Your original table and its rows are kept. New columns are added automatically.

Command line (run from the project folder):
  python database.py import data/recommendations.csv
"""

import csv
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from labels import crop_name, is_healthy

TEXT_FIELDS = ("symptoms", "treatment", "prevention", "nutrients")

# Your original two example rows (kept exactly as they were).
ORIGINAL_ROWS = [
    (
        "Tomato___Early_blight", "Tomato",
        "Possible signs include dark leaf spots, sometimes with concentric rings.",
        "This is general information, not a diagnosis. Confirm the disease with a local agricultural expert before choosing a treatment.",
        "Monitor the crop, avoid unnecessary leaf wetness, and remove affected plant material where locally recommended.",
        "Base fertilizer decisions on crop stage and soil-test results; disease detection alone does not determine fertilizer needs.",
    ),
    (
        "Tomato___healthy", "Tomato",
        "The model did not identify a supported disease class in this image.",
        "No treatment is indicated from this prediction alone. Continue monitoring and seek expert advice if symptoms remain.",
        "Use clean tools and inspect leaves, stems, and fruit regularly.",
        "Follow crop-specific nutrient guidance and soil-test results.",
    ),
]

# Shown when a class has no stored text yet. These are deliberately cautious and
# make NO treatment or fertilizer claim.
FALLBACK_TEXT = {
    "symptoms": "No symptom description has been added for this class yet. Compare the leaf with trusted references.",
    "treatment": "No verified treatment is stored for this class yet. Consult a local agricultural officer or qualified agronomist.",
    "prevention": "Monitor the crop and follow locally approved crop-hygiene practices.",
    "nutrients": "Use crop-specific, soil-test-based fertilizer guidance. Do not infer fertilizer need from the image prediction alone.",
}
HEALTHY_FALLBACK_TEXT = {
    "symptoms": "The model did not identify a supported disease class in this image.",
    "treatment": "No treatment is indicated from this prediction alone. Continue monitoring and seek expert advice if symptoms remain.",
    "prevention": "Use clean tools and inspect leaves, stems, and fruit regularly.",
    "nutrients": "Follow crop-specific nutrient guidance and soil-test results.",
}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def db_session(db_path):
    """Open the database, commit on success, ALWAYS close (important on Windows)."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------- set-up
def init_database(db_path, class_names=()):
    """Create / upgrade tables. Safe to run every time the app starts."""
    with db_session(db_path) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                disease TEXT PRIMARY KEY,
                crop TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                treatment TEXT NOT NULL,
                prevention TEXT NOT NULL,
                nutrients TEXT NOT NULL
            )
        """)
        existing = {row["name"] for row in db.execute("PRAGMA table_info(recommendations)")}
        if "source" not in existing:
            db.execute("ALTER TABLE recommendations ADD COLUMN source TEXT NOT NULL DEFAULT ''")
        if "verified" not in existing:
            db.execute("ALTER TABLE recommendations ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")
        if "updated_at" not in existing:
            db.execute("ALTER TABLE recommendations ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")

        db.executemany(
            """INSERT OR IGNORE INTO recommendations
               (disease, crop, symptoms, treatment, prevention, nutrients)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ORIGINAL_ROWS,
        )
        db.execute(
            """UPDATE recommendations SET source = 'Original project data (general guidance)'
               WHERE source = '' AND disease IN (?, ?)""",
            (ORIGINAL_ROWS[0][0], ORIGINAL_ROWS[1][0]),
        )

        # One row for every class the model can predict, so nothing is ever "missing".
        db.executemany(
            """INSERT OR IGNORE INTO recommendations
               (disease, crop, symptoms, treatment, prevention, nutrients)
               VALUES (?, ?, '', '', '', '')""",
            [(name, crop_name(name)) for name in class_names],
        )

        db.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_file TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                top_json TEXT NOT NULL,
                low_confidence INTEGER NOT NULL DEFAULT 0
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at)")


# ------------------------------------------------------------- recommendations
def get_recommendation(db_path, disease):
    """Return the recommendation for a class. Empty fields get a cautious fallback."""
    with db_session(db_path) as db:
        row = db.execute("SELECT * FROM recommendations WHERE disease = ?", (disease,)).fetchone()
    data = dict(row) if row else {}
    fallback = HEALTHY_FALLBACK_TEXT if is_healthy(disease) else FALLBACK_TEXT
    has_details = any((data.get(f) or "").strip() for f in TEXT_FIELDS)
    result = {
        "disease": disease,
        "crop": data.get("crop") or crop_name(disease),
        "source": data.get("source", ""),
        "verified": bool(data.get("verified", 0)),
        "has_details": has_details,
    }
    for field in TEXT_FIELDS:
        text = (data.get(field) or "").strip()
        result[field] = text if text else fallback[field]
    return result


def import_csv(db_path, csv_path, valid_classes=None):
    """Load recommendations from a CSV file (UTF-8). Returns (imported, skipped_names).

    Columns: disease, crop, symptoms, treatment, prevention, nutrients, source, verified
    Empty cells never erase text that is already in the database.
    """
    imported, skipped = 0, []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f, db_session(db_path) as db:
        for row in csv.DictReader(f):
            disease = (row.get("disease") or "").strip()
            if not disease:
                continue
            if valid_classes is not None and disease not in valid_classes:
                skipped.append(disease)
                continue
            source = (row.get("source") or "").strip()
            verified = 1 if (row.get("verified") or "").strip().lower() in ("1", "yes", "true") else 0
            if verified and not source:
                verified = 0  # "verified" only makes sense when a source is written down
            db.execute(
                """
                INSERT INTO recommendations
                    (disease, crop, symptoms, treatment, prevention, nutrients, source, verified, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(disease) DO UPDATE SET
                    crop       = COALESCE(NULLIF(excluded.crop, ''), crop),
                    symptoms   = COALESCE(NULLIF(excluded.symptoms, ''), symptoms),
                    treatment  = COALESCE(NULLIF(excluded.treatment, ''), treatment),
                    prevention = COALESCE(NULLIF(excluded.prevention, ''), prevention),
                    nutrients  = COALESCE(NULLIF(excluded.nutrients, ''), nutrients),
                    source     = CASE WHEN excluded.source != '' THEN excluded.source ELSE source END,
                    verified   = CASE WHEN excluded.source != '' THEN excluded.verified ELSE verified END,
                    updated_at = excluded.updated_at
                """,
                (
                    disease,
                    (row.get("crop") or "").strip() or crop_name(disease),
                    (row.get("symptoms") or "").strip(),
                    (row.get("treatment") or "").strip(),
                    (row.get("prevention") or "").strip(),
                    (row.get("nutrients") or "").strip(),
                    source,
                    verified,
                    utc_now(),
                ),
            )
            imported += 1
    return imported, skipped


# ---------------------------------------------------------------------- scans
def record_scan(db_path, image_file, result):
    with db_session(db_path) as db:
        cursor = db.execute(
            """INSERT INTO scans (created_at, image_file, prediction, confidence, top_json, low_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                utc_now(),
                image_file,
                result["label"],
                result["confidence"],
                json.dumps(result["top"]),
                1 if result["low_confidence"] else 0,
            ),
        )
        return cursor.lastrowid


def get_scan(db_path, scan_id):
    with db_session(db_path) as db:
        row = db.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return None
    scan = dict(row)
    scan["top"] = json.loads(scan.pop("top_json"))
    scan["low_confidence"] = bool(scan["low_confidence"])
    return scan


def get_stats(db_path):
    """All numbers shown on the dashboard. Everything comes from real stored scans."""
    with db_session(db_path) as db:
        total = db.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        avg = db.execute("SELECT AVG(confidence) FROM scans").fetchone()[0]
        low = db.execute("SELECT COUNT(*) FROM scans WHERE low_confidence = 1").fetchone()[0]
        healthy = db.execute(
            "SELECT COUNT(*) FROM scans WHERE LOWER(prediction) LIKE '%\\_\\_\\_healthy' ESCAPE '\\'"
        ).fetchone()[0]
        top = [
            dict(r) for r in db.execute(
                """SELECT prediction, COUNT(*) AS n FROM scans
                   GROUP BY prediction ORDER BY n DESC, prediction LIMIT 6"""
            )
        ]
        recent = [
            dict(r) for r in db.execute(
                """SELECT id, created_at, prediction, confidence, low_confidence
                   FROM scans ORDER BY id DESC LIMIT 8"""
            )
        ]
        classes_total = db.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
        with_details = db.execute(
            """SELECT COUNT(*) FROM recommendations
               WHERE symptoms != '' OR treatment != '' OR prevention != '' OR nutrients != ''"""
        ).fetchone()[0]
        verified = db.execute("SELECT COUNT(*) FROM recommendations WHERE verified = 1").fetchone()[0]
    return {
        "total": total,
        "healthy": healthy,
        "diseased": total - healthy,
        "low_confidence": low,
        "avg_confidence": round(avg, 1) if avg is not None else None,
        "top": top,
        "recent": recent,
        "classes_total": classes_total,
        "classes_with_details": with_details,
        "classes_verified": verified,
    }


if __name__ == "__main__":
    from predictor import read_class_names

    base = Path(__file__).resolve().parent
    db_file = base / "crop_data.db"
    classes = read_class_names(base / "model" / "class_names.json")
    init_database(db_file, classes)
    if len(sys.argv) >= 3 and sys.argv[1] == "import":
        count, skipped = import_csv(db_file, sys.argv[2], valid_classes=set(classes))
        print(f"Imported {count} rows.")
        if skipped:
            print("Skipped (name is not in class_names.json):")
            for name in skipped:
                print("  -", name)
    else:
        print("Database is ready. To load a CSV:  python database.py import data/recommendations.csv")
