"""Prediction helpers that preserve the fitted model's feature order and pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def predict_profiles(
    profiles: pd.DataFrame, bundle: dict[str, Any]
) -> list[dict[str, float | int | None]]:
    """Predict using the saved estimator; include probability only when available."""
    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in profiles.columns]
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(missing)}")

    model = bundle["model"]
    ordered_profiles = profiles.loc[:, feature_columns]
    predict_proba = getattr(model, "predict_proba", None)
    classes = list(getattr(model, "classes_", []))

    if callable(predict_proba) and 1 in classes:
        probabilities = np.asarray(predict_proba(ordered_profiles))
        positive_index = classes.index(1)
        if probabilities.ndim != 2 or positive_index >= probabilities.shape[1]:
            raise ValueError("The saved model returned an unsupported probability shape.")
        threshold = float(
            bundle["results"]["Stacking Classifier"]["decision_threshold"]
        )
        return [
            {
                "class": int(probability >= threshold),
                "probability": float(probability),
                "threshold": threshold,
            }
            for probability in probabilities[:, positive_index]
        ]

    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("The saved model does not support class predictions.")
    predictions = np.asarray(predict(ordered_profiles))
    return [
        {"class": int(prediction), "probability": None, "threshold": None}
        for prediction in predictions
    ]


def validate_uploaded_profiles(
    uploaded: pd.DataFrame,
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Validate upload schema and dataset-observed feature ranges/categories."""
    missing = [column for column in feature_columns if column not in uploaded.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if uploaded.empty:
        raise ValueError("The uploaded CSV must contain at least one patient record.")
    profiles = uploaded.loc[:, feature_columns].copy()

    for column in feature_columns:
        if column not in dataset.columns:
            raise ValueError(f"Dataset is missing model feature '{column}'.")
        observed = dataset[column]
        if pd.api.types.is_numeric_dtype(observed):
            values = pd.to_numeric(profiles[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"'{column}' must contain finite numeric values.")
            lower = float(pd.to_numeric(observed, errors="coerce").min())
            upper = float(pd.to_numeric(observed, errors="coerce").max())
            if ((values < lower) | (values > upper)).any():
                raise ValueError(
                    f"'{column}' values must be within the observed range "
                    f"{lower:g}–{upper:g}."
                )
            observed_values = set(pd.to_numeric(observed, errors="coerce").dropna().unique())
            if observed_values.issubset({0, 1}) and not values.isin(observed_values).all():
                raise ValueError(f"'{column}' must use one of the observed values {sorted(observed_values)}.")
            profiles[column] = values
        else:
            if profiles[column].isna().any():
                raise ValueError(f"'{column}' cannot contain empty values.")
            valid_values = set(observed.dropna().astype(str).unique())
            invalid = sorted(set(profiles[column].astype(str)) - valid_values)
            if invalid:
                raise ValueError(
                    f"'{column}' contains unsupported categories: {', '.join(invalid)}."
                )

    return profiles.loc[:, feature_columns]
