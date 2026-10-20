import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from heart_monitoring import SCENARIOS, generate_ecg_window, simulated_bpm
from model_predictions import predict_profiles, validate_uploaded_profiles
from synthetic_patients import generate_synthetic_profiles


class PredictionFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = pd.read_csv(PROJECT_DIR / "data" / "full_data.csv")
        cls.bundle = joblib.load(PROJECT_DIR / "models" / "stroke_model.joblib")

    def test_synthetic_profiles_use_saved_feature_order_and_observed_values(self):
        features = self.bundle["feature_columns"]
        generated = generate_synthetic_profiles(
            self.dataset, features, self.bundle["numeric_columns"], 100, random_state=12
        )
        self.assertEqual(list(generated.columns), features)
        self.assertEqual(len(generated), 100)
        self.assertNotIn("stroke", generated.columns)
        for column in self.dataset.select_dtypes(exclude=np.number).columns:
            if column in features:
                self.assertTrue(set(generated[column]).issubset(set(self.dataset[column])))
        for column in self.bundle["numeric_columns"]:
            self.assertGreaterEqual(generated[column].min(), self.dataset[column].min())
            self.assertLessEqual(generated[column].max(), self.dataset[column].max())

    def test_saved_model_predicts_using_exact_feature_order(self):
        profiles = self.dataset.iloc[:3][self.bundle["feature_columns"]]
        predictions = predict_profiles(profiles, self.bundle)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(item["class"] in {0, 1} for item in predictions))
        self.assertTrue(all(0.0 <= item["probability"] <= 1.0 for item in predictions))

    def test_upload_validation_reorders_and_drops_unused_columns(self):
        features = self.bundle["feature_columns"]
        upload = self.dataset.iloc[:2][features[::-1]].copy()
        upload["stroke"] = self.dataset.iloc[:2]["stroke"]
        checked = validate_uploaded_profiles(upload, self.dataset, features)
        self.assertEqual(list(checked.columns), features)
        self.assertEqual(len(checked), 2)

    def test_upload_validation_rejects_out_of_range_and_unknown_categories(self):
        features = self.bundle["feature_columns"]
        upload = self.dataset.iloc[:1][features].copy()
        upload.loc[upload.index[0], "age"] = self.dataset["age"].max() + 1
        with self.assertRaisesRegex(ValueError, "observed range"):
            validate_uploaded_profiles(upload, self.dataset, features)

        upload = self.dataset.iloc[:1][features].copy()
        upload.loc[upload.index[0], "gender"] = "Unknown category"
        with self.assertRaisesRegex(ValueError, "unsupported categories"):
            validate_uploaded_profiles(upload, self.dataset, features)

    def test_ecg_demo_returns_finite_waveform_and_scenario_bpm(self):
        for scenario, (_, low, high) in SCENARIOS.items():
            bpm = simulated_bpm(scenario, elapsed_seconds=10.0)
            waveform = generate_ecg_window(scenario, elapsed_seconds=10.0)
            self.assertGreaterEqual(bpm, low)
            self.assertLessEqual(bpm, high)
            self.assertEqual(len(waveform), 250 * 8)
            self.assertTrue(np.isfinite(waveform["voltage"]).all())


if __name__ == "__main__":
    unittest.main()
