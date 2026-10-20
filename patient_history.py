"""Local SQLite storage for explicitly saved educational prediction sessions."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parent / "data" / "patient_history.sqlite3"


@contextmanager
def _database_connection(
    database_path: Path | str,
) -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(database_path, timeout=10)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_history_database(
    database_path: Path | str = DATABASE_PATH,
) -> None:
    """Create the local history database and table when they do not exist."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _database_connection(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                patient_name TEXT NOT NULL DEFAULT '',
                patient_features_json TEXT NOT NULL,
                predicted_class INTEGER NOT NULL CHECK (predicted_class IN (0, 1)),
                model_probability REAL,
                simulated_bpm REAL,
                simulated_readings_json TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(prediction_history)"
            ).fetchall()
        }
        if "patient_name" not in columns:
            connection.execute(
                "ALTER TABLE prediction_history "
                "ADD COLUMN patient_name TEXT NOT NULL DEFAULT ''"
            )


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported history value type: {type(value).__name__}")


def save_history_record(
    patient_features: dict[str, Any],
    model_output: dict[str, Any],
    *,
    patient_name: str = "",
    simulated_bpm: float | None = None,
    simulated_readings: list[dict[str, Any]] | None = None,
    database_path: Path | str = DATABASE_PATH,
) -> int:
    """Save one prediction session and return its local record ID."""
    if not isinstance(patient_features, dict) or not patient_features:
        raise ValueError("Patient features must be a non-empty mapping.")
    if not isinstance(patient_name, str):
        raise ValueError("Patient name must be text.")
    patient_name = patient_name.strip()
    predicted_class = model_output.get("class")
    if predicted_class not in (0, 1):
        raise ValueError("Predicted class must be 0 or 1.")
    probability = model_output.get("probability")
    if probability is not None:
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Model probability must be between 0 and 1.")
    if simulated_bpm is not None:
        simulated_bpm = float(simulated_bpm)
        if not math.isfinite(simulated_bpm) or simulated_bpm <= 0:
            raise ValueError("Simulated BPM must be a finite positive number.")
    if simulated_readings is None:
        simulated_readings = []
    if not isinstance(simulated_readings, list):
        raise ValueError("Simulated readings must be a list.")

    initialize_history_database(database_path)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    features_json = json.dumps(_json_safe(patient_features), allow_nan=False)
    readings_json = json.dumps(_json_safe(simulated_readings), allow_nan=False)
    with _database_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                created_at, patient_name, patient_features_json, predicted_class,
                model_probability, simulated_bpm, simulated_readings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                patient_name,
                features_json,
                int(predicted_class),
                probability,
                simulated_bpm,
                readings_json,
            ),
        )
        return int(cursor.lastrowid)


def list_history_records(
    database_path: Path | str = DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Return decoded history records, newest first."""
    initialize_history_database(database_path)
    with _database_connection(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, created_at, patient_features_json, predicted_class,
                   model_probability, simulated_bpm, simulated_readings_json,
                   patient_name
            FROM prediction_history
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "patient_name": row["patient_name"],
            "patient_features": json.loads(row["patient_features_json"]),
            "predicted_class": row["predicted_class"],
            "model_probability": row["model_probability"],
            "simulated_bpm": row["simulated_bpm"],
            "simulated_readings": json.loads(row["simulated_readings_json"]),
        }
        for row in rows
    ]


def get_history_record(
    record_id: int,
    database_path: Path | str = DATABASE_PATH,
) -> dict[str, Any] | None:
    """Return one decoded record by ID, or None if it was deleted/not found."""
    return next(
        (
            record
            for record in list_history_records(database_path)
            if record["id"] == record_id
        ),
        None,
    )


def delete_history_record(
    record_id: int,
    database_path: Path | str = DATABASE_PATH,
) -> bool:
    """Delete one record and report whether it existed."""
    initialize_history_database(database_path)
    with _database_connection(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM prediction_history WHERE id = ?", (record_id,)
        )
        return cursor.rowcount > 0


def clear_history(database_path: Path | str = DATABASE_PATH) -> int:
    """Delete all records and return the number removed."""
    initialize_history_database(database_path)
    with _database_connection(database_path) as connection:
        cursor = connection.execute("DELETE FROM prediction_history")
        return cursor.rowcount
