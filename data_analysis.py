"""Print a factual schema, missing-value, and class-balance report for the CSV."""

from __future__ import annotations

import json

from train_model import DATA_PATH, load_dataset


def main() -> None:
    frame, target, summary = load_dataset(DATA_PATH)
    print("DATASET:", DATA_PATH)
    print(f"Records: {len(frame):,}")
    print(f"Predictor columns: {frame.shape[1]}")
    print("Columns:", ", ".join(frame.columns))
    print("\nMissing values:")
    print(frame.isna().sum().to_string())
    print("\nTarget distribution:")
    print(target.value_counts().sort_index().rename("records").to_string())
    print("\nTarget percentages:")
    print((target.value_counts(normalize=True).sort_index() * 100).round(2).to_string())
    print("\nNumeric descriptive statistics:")
    print(frame.describe(include="number").round(2).to_string())
    print("\nCategorical descriptive statistics:")
    categorical = frame.describe(include=["object", "category"])
    print(categorical.to_string() if not categorical.empty else "No categorical columns.")
    print("\nMachine-readable summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
