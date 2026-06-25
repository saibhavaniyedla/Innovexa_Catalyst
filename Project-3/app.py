import os
import joblib
import pandas as pd
import streamlit as st
from src.features import add_features

MODEL_PATH = "models/traffic_demand_model.pkl"

st.set_page_config(page_title="Traffic Demand Prediction", page_icon="🚦", layout="wide")

st.title("🚦 Traffic Demand Prediction using ML")
st.write(
    "This Streamlit app predicts expected traffic demand using road, weather, time, "
    "location and event-based inputs. It is inspired by the Innovexa Catalyst ML task."
)

if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Please run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2, col3 = st.columns(3)

with col1:
    junction_location = st.selectbox("Junction Location", [
        "Junction_A", "Junction_B", "Metro_Road", "Market_Street", "Highway_Entry", "School_Zone"
    ])
    day_of_week = st.selectbox("Day of Week", [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])
    hour = st.slider("Hour", 0, 23, 9)

with col2:
    road_type = st.selectbox("Road Type", ["Urban", "Highway", "Residential", "Commercial"])
    number_of_lanes = st.slider("Number of Lanes", 1, 6, 3)
    traffic_signals_count = st.slider("Traffic Signals Count", 0, 5, 2)

with col3:
    weather_conditions = st.selectbox("Weather Conditions", ["Clear", "Cloudy", "Rainy", "Foggy"])
    temperature = st.number_input("Temperature", value=30.0)
    humidity = st.number_input("Humidity", value=65.0)
    rainfall = st.number_input("Rainfall", min_value=0.0, value=0.0)
    large_vehicles_count = st.slider("Large Vehicles Count", 0, 100, 15)
    nearby_landmarks = st.slider("Nearby Landmarks", 0, 5, 2)
    event_indicator = st.selectbox("Event Indicator", [0, 1], format_func=lambda x: "Yes" if x else "No")

if st.button("Predict Traffic Demand"):
    input_df = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "junction_location": junction_location,
        "day_of_week": day_of_week,
        "hour": hour,
        "road_type": road_type,
        "number_of_lanes": number_of_lanes,
        "traffic_signals_count": traffic_signals_count,
        "large_vehicles_count": large_vehicles_count,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "weather_conditions": weather_conditions,
        "nearby_landmarks": nearby_landmarks,
        "event_indicator": event_indicator
    }])

    input_df = add_features(input_df)
    X = input_df.drop(columns=["timestamp"])
    prediction = int(model.predict(X)[0])

    if prediction >= 3000:
        congestion = "High"
    elif prediction >= 1800:
        congestion = "Medium"
    else:
        congestion = "Low"

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Traffic Demand", f"{prediction:,} vehicles/hour")
    m2.metric("Congestion Level", congestion)
    m3.metric("Peak Traffic Alert", "Yes" if congestion == "High" else "No")

    st.info("Use this output to plan signal timing, route diversion, manpower, and congestion alerts.")
