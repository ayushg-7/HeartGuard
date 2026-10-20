"""Generate clearly synthetic patient profiles from observed dataset values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_profiles(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    count: int,
    *,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Bootstrap feature profiles and perturb numeric values within observed ranges."""
    if count not in {100, 500, 1000}:
        raise ValueError("Synthetic record count must be 100, 500, or 1,000.")
    missing = [column for column in feature_columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {', '.join(missing)}")
    if dataset.empty:
        raise ValueError("Cannot generate synthetic profiles from an empty dataset.")

    rng = np.random.default_rng(random_state)
    profiles = dataset.loc[:, feature_columns].sample(
        n=count,
        replace=True,
        random_state=random_state,
    ).reset_index(drop=True)
    for column in numeric_columns:
        values = pd.to_numeric(dataset[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"Numeric feature '{column}' has no valid dataset values.")
        low = float(values.min())
        high = float(values.max())
        if set(values.unique()).issubset({0, 1}):
            profiles[column] = rng.choice(values.unique(), size=count)
            continue
        spread = float(values.std(ddof=0))
        if spread:
            noise = rng.normal(0.0, spread * 0.03, size=count)
            generated = pd.to_numeric(profiles[column], errors="raise").to_numpy() + noise
            profiles[column] = np.clip(generated, low, high)
        else:
            profiles[column] = low

    return profiles.loc[:, feature_columns]
