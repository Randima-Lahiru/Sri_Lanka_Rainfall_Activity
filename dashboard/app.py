# Import libraries
import streamlit as st
import pandas as pd
import joblib
from datetime import date
from pathlib import Path


# Project folders
dashboard_dir = Path(__file__).parent
project_dir = dashboard_dir.parent


# Load trained model files
model = joblib.load(
    project_dir / "models" / "rain_prediction_model.pkl"
)

features = joblib.load(
    project_dir / "models" / "model_features.pkl"
)


# Load dashboard data files
locations = pd.read_csv(
    dashboard_dir / "locations_dashboard.csv"
)

comparison = pd.read_csv(
    dashboard_dir / "model_comparison.csv"
)

feature_importance = pd.read_csv(
    dashboard_dir / "feature_importance.csv"
)


# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="Sri Lanka Rainfall Prediction",
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ Sri Lanka Rainfall Prediction Dashboard")

st.caption(
    "Random Forest vs XGBoost using Sri Lankan weather data"
)


# ---------------------------------------------------
# Create dashboard tabs
# ---------------------------------------------------

prediction_tab, comparison_tab, importance_tab = st.tabs(
    [
        "Prediction",
        "Model Comparison",
        "Feature Importance"
    ]
)


# ===================================================
# TAB 1 - PREDICTION
# ===================================================

with prediction_tab:

    st.header("Predict Tomorrow's Rainfall")

    st.write(
        "Enter today's main weather conditions "
        "to predict whether it will rain tomorrow."
    )


    # -----------------------------
    # Location and date
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        city = st.selectbox(
            "Location",
            sorted(locations["city_name"].unique())
        )

    with col2:

        prediction_date = st.date_input(
            "Current Date",
            value=date.today()
        )


    # Get location information automatically
    selected_location = locations[
        locations["city_name"] == city
    ].iloc[0]


    # -----------------------------
    # Main weather inputs
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        temperature_max = st.number_input(
            "Maximum Temperature (°C)",
            value=31.0
        )

    with col2:

        temperature_min = st.number_input(
            "Minimum Temperature (°C)",
            value=25.0
        )


    col1, col2 = st.columns(2)

    with col1:

        rain_today = st.number_input(
            "Today's Rainfall (mm)",
            min_value=0.0,
            value=0.0
        )

    with col2:

        precipitation_hours = st.number_input(
            "Precipitation Hours",
            min_value=0.0,
            value=0.0
        )


    wind_speed = st.number_input(
        "Maximum Wind Speed (km/h)",
        min_value=0.0,
        value=15.0
    )


    # -----------------------------
    # Make prediction
    # -----------------------------

    if st.button("Predict Tomorrow's Rain"):

        # Derive a few supporting values
        temperature_mean = (
            temperature_max + temperature_min
        ) / 2

        apparent_temperature = temperature_mean

        # Simple demonstration values
        sunshine_duration = 22000.0
        wind_gust = wind_speed * 1.5
        wind_direction = 180.0
        radiation = 15.0
        evapotranspiration = 3.0


        # Prepare input for the model
        input_data = pd.DataFrame([{

            "temperature_2m_max (°C)":
                temperature_max,

            "temperature_2m_min (°C)":
                temperature_min,

            "temperature_2m_mean (°C)":
                temperature_mean,

            "apparent_temperature_mean (°C)":
                apparent_temperature,

            "sunshine_duration (s)":
                sunshine_duration,

            "rain_sum (mm)":
                rain_today,

            "precipitation_hours (h)":
                precipitation_hours,

            "wind_speed_10m_max (km/h)":
                wind_speed,

            "wind_gusts_10m_max (km/h)":
                wind_gust,

            "wind_direction_10m_dominant (°)":
                wind_direction,

            "shortwave_radiation_sum (MJ/m²)":
                radiation,

            "et0_fao_evapotranspiration (mm)":
                evapotranspiration,

            "month":
                prediction_date.month,

            "day_of_year":
                prediction_date.timetuple().tm_yday,

            "latitude":
                selected_location["latitude"],

            "longitude":
                selected_location["longitude"],

            "elevation":
                selected_location["elevation"]
        }])


        # Keep the same feature order used during training
        input_data = input_data[features]


        # Make prediction
        prediction = model.predict(input_data)[0]

        rain_probability = model.predict_proba(
            input_data
        )[0][1]


        # -----------------------------
        # Show result
        # -----------------------------

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.warning(
                "🌧️ Rain is likely tomorrow"
            )

        else:

            st.success(
                "☀️ No significant rain is predicted tomorrow"
            )


        st.metric(
            "Rain Probability",
            f"{rain_probability * 100:.1f}%"
        )


# ===================================================
# TAB 2 - MODEL COMPARISON
# ===================================================

with comparison_tab:

    st.header(
        "Random Forest vs XGBoost"
    )

    st.write(
        "Performance comparison using the same test dataset."
    )


    # Show results table
    st.dataframe(
        comparison,
        use_container_width=True
    )


    # Prepare data for chart
    chart_data = comparison.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]


    # Show comparison chart
    st.bar_chart(
        chart_data
    )


    # Find best model using F1 Score
    best_row = comparison.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]


    st.success(
        f"Best Model: {best_row['Model']} "
        f"(F1 Score: {best_row['F1 Score']:.4f})"
    )


# ===================================================
# TAB 3 - FEATURE IMPORTANCE
# ===================================================

with importance_tab:

    st.header(
        "Feature Importance"
    )

    st.write(
        "Top features used by the selected model."
    )


    # Select top 10 features
    top_features = (
        feature_importance
        .head(10)
    )


    # Show chart
    st.bar_chart(
        top_features.set_index("Feature")
    )


    # Show table
    st.dataframe(
        top_features,
        use_container_width=True
    )