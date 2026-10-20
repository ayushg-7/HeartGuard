# NeuroPulse / HeartGuard Project Overview

## Project title

**AI-Based Stroke Prediction with Simulated ECG Monitoring**

The current dashboard branding is **HeartGuard**. This project is an educational demonstration; its model predictions are experimental and its BPM/ECG-style signals are simulated. It is not a clinical tool or medical record.

## Abstract

This project provides an interactive Streamlit dashboard for exploring a supplied stroke dataset and a saved stacking machine-learning model. Users can enter patient features, view an experimental predicted class and model probability when available, and review model evaluation results and dataset visualizations. A separate heart-monitoring screen demonstrates simulated BPM readings and an ECG-style waveform. Patient prediction sessions can be saved explicitly to a local SQLite database, reviewed, searched, filtered, and exported. The simulation is independent of the prediction model, and neither output is intended for medical decision-making.

## Problem statement

Exploring a machine-learning prediction workflow through scripts alone can make it difficult to understand how patient inputs, model output, and evaluation results fit together. This project addresses that educational need with an interactive dashboard for entering supported dataset features, viewing the saved model's output, and exploring the supplied data and model performance.

The dashboard also demonstrates a simulated heart-rate waveform, but it must clearly distinguish simulation from real measurement and avoid presenting predictions or BPM values as diagnoses.

## Proposed solution

Build an interactive Streamlit dashboard that:

- Accepts patient features supported by the supplied dataset and validates them against observed categories and numeric ranges.
- Runs the existing saved stacking model and displays its predicted class and probability when supported.
- Presents model evaluation metrics and exploratory views of the data.
- Demonstrates a separate software-generated BPM and ECG-style waveform.
- Allows a user to explicitly save patient details and prediction output to local SQLite history, then search, review, export, or delete saved sessions.

The patient's name is stored with a saved history session for identification; it is not passed to the prediction model. Simulated BPM is also kept separate from the model inputs.

## Dashboard features

### Home Dashboard

Provides a high-level view of the project, dataset, and model status.

### Heart Stroke Prediction

Accepts patient name and supported patient features, runs the saved model, and shows the experimental class and probability when available. The user can explicitly save the result to history.

### Heart Monitoring

Shows a software-generated BPM and ECG-style waveform. The user can choose a demo scenario, enter a simulated BPM, start/pause/reset monitoring, and configure a demo range alert. No physical ECG sensor is connected.

### Patient Health History

Provides a patient-entry form on the page, prediction, and explicit save to local history. Saved sessions can be searched/filtered, opened for details, exported as CSV, or deleted after confirmation. Existing history remains available after restarting the app.

### Exploratory Data Analysis

Shows visual summaries of selected patient features and dataset patterns.

### Model Comparison

Displays evaluation results for the saved model and comparison classifiers, including relevant classification metrics and plots.

## Architecture / block diagram

```text
                         ┌───────────────────────┐
                         │      User / Viewer    │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  Streamlit Dashboard  │
                         └──────┬──────────┬─────┘
                                │          │
                  Patient name + features │ Monitoring controls
                                │          │
                    ┌───────────▼────┐  ┌──▼────────────────────┐
                    │ Input form and │  │ Simulated BPM + ECG   │
                    │ validation     │  │ waveform (demo only)  │
                    └───────────┬────┘  └───────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Saved stacking model   │
                    │ (patient features only)│
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │ Experimental class and │
                    │ probability, if any   │
                    └───────────┬────────────┘
                                │ User explicitly saves
                    ┌───────────▼────────────┐
                    │ Local SQLite history   │
                    │ search / detail / CSV  │
                    │ export / deletion      │
                    └────────────────────────┘

 Supplied dataset ──► model training/evaluation ──► saved model artifact
        └───────────► exploratory data analysis
```

## Model workflow

```text
Patient features
      │
      ▼
Input validation and preprocessing
      │
      ▼
Saved stacking classifier
      │
      ├──► Predicted class
      └──► Model probability (when available)
```

The supplied model uses the available dataset features, including age, gender, hypertension, heart disease, glucose level, BMI, and recorded categories. Training and evaluation are based on the supplied dataset. The app loads the saved model artifact; it does not retrain the model on every launch.

## Data and safety notes

- **Model results are experimental** and should not be interpreted as a diagnosis or used to make treatment decisions.
- **BPM and ECG-style signals are simulated**, not measured from a person.
- The simulated monitoring data is not used by the stroke prediction model.
- History is saved locally in `data/patient_history.sqlite3` only when the user explicitly chooses a save action.
- The local SQLite file is not encrypted. Use only suitable demonstration data and protect access to the computer.

## Run the dashboard

From the `stroke-prediction` project folder in PowerShell:

```powershell
venv\Scripts\activate
streamlit run app.py
```

The app uses the existing saved model in `models\stroke_model.joblib`. Do not run model training unless the artifact is missing or retraining is specifically intended.
