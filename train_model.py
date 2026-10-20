"""Train and evaluate stroke prediction models on the supplied dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "full_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "stroke_model.joblib"
SUMMARY_PATH = MODEL_DIR / "dataset_summary.json"
TARGET_COLUMN = "stroke"
RANDOM_STATE = 42
IDENTIFIER_COLUMNS = {"id", "patient_id", "patientid", "record_id", "recordid"}


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Read the CSV, remove known identifiers, and report observed data facts."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}'. Found columns: {list(frame.columns)}"
        )

    identifiers = [
        column for column in frame.columns
        if column.strip().lower() in IDENTIFIER_COLUMNS
    ]
    frame = frame.drop(columns=identifiers)
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    target = frame.pop(TARGET_COLUMN)
    if target.nunique() != 2:
        raise ValueError(f"'{TARGET_COLUMN}' must contain exactly two classes.")

    observed = set(pd.unique(target))
    if observed != {0, 1}:
        raise ValueError(
            f"'{TARGET_COLUMN}' must be encoded as 0 and 1; observed values: {sorted(observed)}"
        )
    target = target.astype(int)
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "feature_count": int(frame.shape[1]),
        "feature_columns": list(frame.columns),
        "removed_identifier_columns": identifiers,
        "missing_values": {name: int(value) for name, value in frame.isna().sum().items()},
        "target_column": TARGET_COLUMN,
        "class_counts": {
            str(int(label)): int(count) for label, count in target.value_counts().sort_index().items()
        },
        "class_percentages": {
            str(int(label)): float(value)
            for label, value in (target.value_counts(normalize=True).sort_index() * 100).items()
        },
        "categorical_values": {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in frame.select_dtypes(exclude=np.number).columns
        },
    }
    return frame, target, summary


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    """Impute, scale, and encode within each model's fitted training pipeline."""
    numeric_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def build_model(
    estimator: Any, numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    """Build a leak-safe pipeline; resampling runs only during fit."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
            ("oversampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("classifier", estimator),
        ]
    )


def create_models(
    numeric_columns: list[str], categorical_columns: list[str]
) -> dict[str, Any]:
    """Create individual baselines and a stacking ensemble from the same base learners."""
    logistic = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    forest = RandomForestClassifier(
        n_estimators=160, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
    )
    gradient_boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)

    base_estimators = [
        (
            "logistic_regression",
            build_model(
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                numeric_columns,
                categorical_columns,
            ),
        ),
        (
            "random_forest",
            build_model(
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
                numeric_columns,
                categorical_columns,
            ),
        ),
        (
            "gradient_boosting",
            build_model(
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                numeric_columns,
                categorical_columns,
            ),
        ),
    ]
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    return {
        "Logistic Regression": build_model(logistic, numeric_columns, categorical_columns),
        "Random Forest": build_model(forest, numeric_columns, categorical_columns),
        "Gradient Boosting": build_model(
            gradient_boosting, numeric_columns, categorical_columns
        ),
        "Stacking Classifier": stack,
    }


def select_f1_threshold(
    model: Any, X_validation: pd.DataFrame, y_validation: pd.Series
) -> float:
    """Choose an F1-maximizing probability cutoff on training-only validation data."""
    probabilities = model.predict_proba(X_validation)[:, 1]
    precision_values, recall_values, thresholds = precision_recall_curve(
        y_validation, probabilities
    )
    if len(thresholds) == 0:
        return 0.5
    denominator = precision_values[:-1] + recall_values[:-1]
    f1_values = np.divide(
        2 * precision_values[:-1] * recall_values[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_indexes = np.flatnonzero(f1_values == f1_values.max())
    return float(thresholds[int(best_indexes[-1])])


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float,
) -> dict[str, Any]:
    """Calculate holdout metrics and plot-ready curve/confusion-matrix values."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_test, probabilities
    )
    return {
        "metrics": {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(precision_score(y_test, predictions, zero_division=0)),
            "Recall": float(recall_score(y_test, predictions, zero_division=0)),
            "F1-score": float(f1_score(y_test, predictions, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probabilities)),
            "PR-AUC": float(average_precision_score(y_test, probabilities)),
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "decision_threshold": float(decision_threshold),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def train_and_save() -> dict[str, Any]:
    """Train all models on the supplied data and persist actual holdout results."""
    X, y, summary = load_dataset()
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    if y_test.nunique() != 2:
        raise ValueError("The stratified test split must contain both target classes.")

    models = create_models(numeric_columns, categorical_columns)
    results: dict[str, Any] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        X_fit, X_threshold, y_fit, y_threshold = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
        model.fit(X_fit, y_fit)
        threshold = select_f1_threshold(model, X_threshold, y_threshold)
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test, threshold)
        metrics_text = ", ".join(
            f"{metric}={value:.3f}"
            for metric, value in results[name]["metrics"].items()
        )
        print(
            f"  {metrics_text}, threshold={threshold:.3f} "
            "(selected on training-only validation data)"
        )

    summary.update(
        {
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "random_state": RANDOM_STATE,
            "test_size": 0.20,
            "threshold_selection": "F1-maximizing threshold on a stratified 20% slice of training data",
        }
    )
    bundle = {
        "model": models["Stacking Classifier"],
        "models": models,
        "results": results,
        "summary": summary,
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "test_target": y_test.astype(int).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved trained model and test results to: {MODEL_PATH}")
    print(f"Saved dataset summary to: {SUMMARY_PATH}")
    return bundle


if __name__ == "__main__":
    train_and_save()
