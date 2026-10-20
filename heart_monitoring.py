"""Synthetic ECG-like waveform and heart-rate utilities for the Streamlit demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIOS: dict[str, tuple[float, float, float]] = {
    "Normal-range demo": (76.0, 60.0, 100.0),
    "Elevated-rate demo": (118.0, 105.0, 140.0),
    "Low-rate demo": (50.0, 40.0, 59.0),
}
SAMPLE_RATE = 250
WINDOW_SECONDS = 8


def simulated_bpm(
    scenario: str,
    elapsed_seconds: float,
    *,
    center_bpm: float | None = None,
) -> float:
    """Return a smoothly varying BPM for a scenario or a manually selected baseline."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown heart-rate scenario: {scenario}")
    center, low, high = SCENARIOS[scenario]
    variation = 3.2 * np.sin(elapsed_seconds * 0.65) + 1.1 * np.sin(
        elapsed_seconds * 0.21
    )
    if center_bpm is not None:
        if not np.isfinite(center_bpm) or not 30 <= center_bpm <= 180:
            raise ValueError("Center BPM must be between 30 and 180.")
        return float(np.clip(center_bpm + variation, 30, 180))
    return float(np.clip(center + variation, low, high))


def generate_ecg_window(
    scenario: str,
    elapsed_seconds: float,
    *,
    bpm: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    window_seconds: int = WINDOW_SECONDS,
) -> pd.DataFrame:
    """Generate a short, synthetic ECG-style trace ending at the elapsed time."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown heart-rate scenario: {scenario}")
    if sample_rate < 1 or window_seconds < 1:
        raise ValueError("Sample rate and window length must be positive.")

    sample_count = sample_rate * window_seconds
    relative_time = np.linspace(
        0.0, float(window_seconds), sample_count, endpoint=False
    )
    absolute_time = max(0.0, elapsed_seconds - window_seconds) + relative_time
    center, _, _ = SCENARIOS[scenario]
    waveform_bpm = center if bpm is None else float(bpm)
    if not np.isfinite(waveform_bpm) or waveform_bpm <= 0:
        raise ValueError("BPM must be a finite positive number.")
    phase = (
        absolute_time * waveform_bpm / 60.0
        + 0.025 * np.sin(absolute_time * 0.55)
    ) % 1.0

    def pulse(center_phase: float, amplitude: float, width: float) -> np.ndarray:
        distance = (phase - center_phase + 0.5) % 1.0 - 0.5
        return amplitude * np.exp(-0.5 * (distance / width) ** 2)

    voltage = (
        pulse(0.18, 0.12, 0.035)
        + pulse(0.36, -0.16, 0.012)
        + pulse(0.39, 1.05, 0.010)
        + pulse(0.42, -0.28, 0.014)
        + pulse(0.68, 0.30, 0.075)
        + 0.012 * np.sin(2.0 * np.pi * relative_time * 0.35)
    )
    return pd.DataFrame(
        {
            "time_seconds": relative_time,
            "voltage": voltage,
        }
    )
