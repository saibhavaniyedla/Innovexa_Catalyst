import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-friendly traffic features from raw traffic records."""
    data = df.copy()
    data["peak_hour_flag"] = data["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    data["weekend_flag"] = data["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    data["rush_hour_indicator"] = data["peak_hour_flag"] * (data["number_of_lanes"] + data["traffic_signals_count"])
    data["weather_impact_score"] = (
        data["weather_conditions"].map({"Clear": 0, "Cloudy": 1, "Foggy": 2, "Rainy": 3}).fillna(1)
        + data["rainfall"].fillna(0)
    )
    data["traffic_density_score"] = (
        data["traffic_signals_count"] * 0.3
        + data["large_vehicles_count"] * 0.2
        + data["nearby_landmarks"] * 0.5
        + data["event_indicator"] * 2
    )
    return data
