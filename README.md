# HeartGuard — Stacking Model for Heart Stroke Prediction using Machine Learning Techniques

A Streamlit educational dashboard that inspects the provided `full_data.csv`, trains three baseline classifiers and a stacking classifier, and displays predictions and holdout evaluation results. It uses only the CSV included in `data/`; it does not fetch or substitute a dataset.

## Dataset inspected

The supplied file contains 4,981 records and 11 columns:

`gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`.

The target column is named `stroke` in the supplied CSV (0 = no heart stroke, 1 = heart stroke); this exact column name is retained in code for compatibility with the data and trained model. This is a heart stroke prediction dataset, not the UCI Cleveland heart-disease dataset. It does not contain Cleveland-only fields such as serum cholesterol, resting blood pressure, maximum heart rate, chest-pain type, or exercise angina; the prediction form therefore uses only the ten predictor columns that actually exist here. The file initially has no missing values and contains 4,733 no-heart-stroke records and 248 heart-stroke records. Training scripts always recalculate and print statistics from the included CSV.

## Project structure

```text
stroke-prediction/
├── app.py
├── patient_history.py
├── train_model.py
├── data_analysis.py
├── heart_monitoring.py
├── requirements.txt
├── README.md
├── data/
│   └── full_data.csv
│   └── patient_history.sqlite3    # created when history is first used
├── models/
│   ├── stroke_model.joblib        # created by training
│   └── dataset_summary.json       # created by training
```

## Windows / VS Code setup

Open the `stroke-prediction` folder in VS Code, then open a PowerShell terminal:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data_analysis.py
streamlit run app.py
```

The saved model is loaded from `models/stroke_model.joblib`; launching or refreshing the app does not retrain it. Run `python train_model.py` only if the model artifact is missing or you intentionally want to retrain. Model metrics and curves are recomputed from a stratified 80/20 split when training is requested; the test portion is never used to fit preprocessing or resampling.

## Methodology

- Identifiers with known non-predictive names are removed; `stroke` is separated as the binary target.
- A stratified 80/20 train/test split uses a fixed seed (42).
- Each model uses a fitted preprocessing pipeline: median imputation and standard scaling for numeric columns, most-frequent imputation and one-hot encoding for categorical columns.
- Random oversampling is fitted only on each training fold inside an imbalanced-learn pipeline. This keeps oversampling out of validation/test data and works for all three base learners.
- Logistic Regression, Random Forest, and Gradient Boosting are evaluated separately and as base estimators in a five-fold `StackingClassifier`. Its Logistic Regression final estimator learns from out-of-fold base-model probabilities.
- For each model, the classification threshold is selected to maximize F1 on a separate stratified validation slice from the training split. The model is then refit on the full training split; the held-out test set is used only for final evaluation.
- Accuracy, precision, recall, F1-score, ROC-AUC, average precision (PR-AUC), confusion matrices, ROC curves, and precision-recall curves are computed on the held-out test set and persisted with the fitted model using Joblib.

With a strongly imbalanced target, accuracy alone can be misleading. Review recall, precision, PR-AUC, and the confusion matrix as well.

## Dashboard pages

- **Home Dashboard:** dataset/model status, key counts, project workflow, and objectives.
- **Heart Stroke Prediction:** validated form and stacking-model probability/result.
- **Live Heart Monitoring:** software-only synthetic ECG-style waveform with normal-range, elevated-rate, and low-rate scenarios; start, pause, reset, recent simulated readings, heart-rate trend, and configurable informational range alerts. Select a dataset row or enter a profile for a side-by-side educational dashboard containing patient features, the saved stacking-model output, and the separate simulated BPM.
- **Patient Health History:** explicitly save prediction sessions locally, search/filter by date and predicted class, view full inputs and model output, confirm individual/all-record deletion, and export selected/all records as CSV. The SQLite database is created automatically; sessions are never saved unless **Save to History** is clicked.
- **Synthetic & Batch Predictions:** generate 100, 500, or 1,000 synthetic profiles from dataset-observed categories and numeric ranges; download profiles and saved-model predictions; or upload a CSV for schema/value validation, batch prediction, and result download.
- **Exploratory Data Analysis:** age, glucose, BMI, risk-group heart-stroke rates, and numeric correlations.
- **Model Comparison:** actual holdout metrics, confusion matrices, ROC/PR curves, and stacking explanation.

The prediction form uses the data's observed category values and dataset-derived numeric bounds. The displayed probability is produced by the fitted classifier when the saved model supports probability estimates. Synthetic records do not contain actual outcomes, and batch predictions are model-generated classes rather than observed results.

Heart Monitoring uses NumPy to generate a synthetic waveform and changing BPM reading, displayed in Plotly. The monitor refreshes every 1.5 seconds only while active. It displays **SIMULATED DATA — NO PHYSICAL EKG SENSOR CONNECTED.** This is not clinical monitoring: simulated BPM cannot detect arrhythmias, diagnose atrial fibrillation, or predict an imminent stroke. Some rhythm abnormalities can be associated with stroke risk, but BPM alone is insufficient to establish such a diagnosis. The saved model is experimental and not a medical diagnosis; the ML and simulation outputs remain separate and are not combined into a clinical score.

Patient Health History stores the entered patient name, explicitly saved feature inputs, experimental model output, and available simulated BPM/readings in a local SQLite database at `data/patient_history.sqlite3`. The name is stored for record identification and is not sent to the prediction model. No record is saved automatically during monitor refreshes. The database is local and not encrypted; treat this as educational demo data, not a clinical record.

Run the focused feature tests from the project folder with:

```powershell
python -m unittest discover -s tests
```
