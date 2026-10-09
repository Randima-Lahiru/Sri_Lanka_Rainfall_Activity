# Sri Lanka Rainfall Prediction Activity

## Overview

This project predicts whether it will rain on the following day using Sri Lankan historical weather data.

The activity compares two machine learning classification models:

- Random Forest
- XGBoost

The models were evaluated using Accuracy, Precision, Recall and F1 Score.
The final selected model is used in a simple Streamlit dashboard.

## Project Structure

```text
Sri_Lanka_Rainfall_Activity/
│
├── notebook/
│   └── Sri_Lanka_Rainfall_Activity.ipynb
│
├── data/
│   ├── weatherData.csv
│   └── locationData.csv
│
├── models/
│   ├── rain_prediction_model.pkl
│   └── model_features.pkl
│
├── dashboard/
│   ├── app.py
│   ├── locations_dashboard.csv
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The project uses Sri Lankan weather data for multiple locations.

The main dataset contains daily weather measurements such as temperature, rainfall, precipitation hours, wind speed, sunshine duration, radiation and evapotranspiration.

The location dataset contains city name, latitude, longitude and elevation.

## Target Variable

The project predicts whether there will be rain on the next day.

- `1` = More than 1 mm of rain tomorrow
- `0` = 1 mm or less rain tomorrow

## Data Split

Because weather observations are time-based, a chronological split was used.

- Training data: before 2022
- Validation data: 2022
- Test data: 2023 onwards

The validation set was used for parameter selection, while the test set was used only for the final model evaluation.

## Models

Two classification models were compared:

### Random Forest

Random Forest was trained using parameters such as number of trees, maximum depth, minimum samples split and minimum samples leaf.

### XGBoost

XGBoost was trained using parameters such as number of estimators, maximum depth, learning rate, subsample and column sample.

A small parameter search was performed using the validation data.

## Final Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.7879 | 0.8260 | 0.8228 | 0.8244 |
| XGBoost | 0.7863 | 0.8273 | 0.8176 | 0.8224 |

Random Forest achieved the highest F1 Score and was selected as the final model.

## Important Features

The most important features in the final model included:

- Rainfall amount
- Precipitation hours
- Evapotranspiration
- Shortwave radiation
- Sunshine duration
- Day of year

## Dashboard

The Streamlit dashboard contains three sections:

1. **Prediction** – predicts whether it is likely to rain tomorrow.
2. **Model Comparison** – compares Random Forest and XGBoost results.
3. **Feature Importance** – displays the most important model features.

The dashboard is intended as a simple demonstration of machine learning inference for this MSc activity.

Some supporting weather values in the dashboard are derived or assigned simple demonstration values when they are not manually entered by the user.

## Installation

Open a terminal inside the project folder.

Create a virtual environment if required:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

From the project root folder:

```bash
streamlit run dashboard/app.py
```

If the `streamlit` command is not available:

```bash
python3 -m streamlit run dashboard/app.py
```

The application will normally open in the browser at:

```text
http://localhost:8501
```

## Notebook

The complete machine learning workflow is available in:

```text
notebook/Sri_Lanka_Rainfall_Activity.ipynb
```

The notebook includes data loading, preprocessing, target creation, feature engineering, train/validation/test splitting, Random Forest, XGBoost, parameter tuning, model evaluation, feature importance and model saving.

## Generated Files

The notebook generates:

```text
rain_prediction_model.pkl
model_features.pkl
model_comparison.csv
feature_importance.csv
locations_dashboard.csv
```

After running the notebook, these files are placed in the `models` and `dashboard` folders according to the project structure.

## Conclusion

This activity demonstrates a simple end-to-end machine learning workflow using Sri Lankan weather data.

Random Forest and XGBoost were trained and compared using the same dataset. Random Forest achieved a slightly higher F1 Score on the final test dataset and was selected as the final model.

The selected model was saved and connected to a Streamlit dashboard to demonstrate rainfall prediction using new input data.
