import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from patient_history import (
    clear_history,
    delete_history_record,
    get_history_record,
    list_history_records,
    save_history_record,
)


class PatientHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "history.sqlite3"

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_saved_records_persist_and_preserve_prediction_and_readings(self):
        record_id = save_history_record(
            {"age": np.int64(62), "bmi": np.nan, "smoking_status": "never smoked"},
            {"class": 1, "probability": 0.73},
            patient_name="  Sample Patient  ",
            simulated_bpm=96,
            simulated_readings=[{"time": "12:00:00", "simulated_bpm": 96.2}],
            database_path=self.database_path,
        )

        record = get_history_record(record_id, database_path=self.database_path)

        self.assertIsNotNone(record)
        self.assertEqual(record["patient_name"], "Sample Patient")
        self.assertEqual(record["patient_features"]["age"], 62)
        self.assertIsNone(record["patient_features"]["bmi"])
        self.assertEqual(record["predicted_class"], 1)
        self.assertEqual(record["model_probability"], 0.73)
        self.assertEqual(record["simulated_bpm"], 96)
        self.assertEqual(record["simulated_readings"][0]["simulated_bpm"], 96.2)
        self.assertEqual(len(list_history_records(self.database_path)), 1)

    def test_probability_and_bpm_can_be_unavailable(self):
        record_id = save_history_record(
            {"age": 45},
            {"class": 0, "probability": None},
            database_path=self.database_path,
        )

        record = get_history_record(record_id, database_path=self.database_path)

        self.assertIsNone(record["model_probability"])
        self.assertIsNone(record["simulated_bpm"])
        self.assertEqual(record["simulated_readings"], [])

    def test_existing_database_is_migrated_with_empty_patient_name(self):
        import sqlite3

        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """
                CREATE TABLE prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    patient_features_json TEXT NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    model_probability REAL,
                    simulated_bpm REAL,
                    simulated_readings_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO prediction_history (
                    created_at, patient_features_json, predicted_class,
                    model_probability, simulated_bpm, simulated_readings_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-10-20T10:00:00+00:00",
                    '{"age": 45}',
                    0,
                    None,
                    None,
                    "[]",
                ),
            )
            connection.commit()
        finally:
            connection.close()

        records = list_history_records(self.database_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["patient_name"], "")
        self.assertEqual(records[0]["patient_features"]["age"], 45)

    def test_delete_one_and_clear_all(self):
        first_id = save_history_record(
            {"age": 45}, {"class": 0}, database_path=self.database_path
        )
        save_history_record(
            {"age": 62}, {"class": 1}, database_path=self.database_path
        )

        self.assertTrue(delete_history_record(first_id, self.database_path))
        self.assertEqual(len(list_history_records(self.database_path)), 1)
        self.assertEqual(clear_history(self.database_path), 1)
        self.assertEqual(list_history_records(self.database_path), [])

    def test_invalid_model_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Predicted class"):
            save_history_record(
                {"age": 45},
                {"class": 2},
                database_path=self.database_path,
            )
        with self.assertRaisesRegex(ValueError, "Model probability"):
            save_history_record(
                {"age": 45},
                {"class": 0, "probability": 1.5},
                database_path=self.database_path,
            )


if __name__ == "__main__":
    unittest.main()
