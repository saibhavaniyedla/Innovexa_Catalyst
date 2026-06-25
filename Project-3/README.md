# Traffic Demand Prediction using LightGBM & XGBoost

## Project Overview
This project builds a machine learning system to predict traffic demand using smart city traffic data. 
The model uses road details, time-based features, weather conditions, nearby landmarks, event indicators, 
and historical traffic patterns to estimate the expected traffic demand.

The project follows the Innovexa Catalyst ML task theme: **Traffic Demand Prediction using LightGBM & XGBoost**.

## Objective
To build a high-accuracy ML system capable of predicting traffic demand and helping city authorities 
understand congestion levels, peak-hour load, and event-based traffic pressure.

## Features Used
- Junction location
- Timestamp / hour
- Day of week
- Road type
- Number of lanes
- Traffic signals count
- Large vehicles count
- Temperature
- Humidity
- Rainfall
- Weather conditions
- Nearby landmarks
- Event indicator

## Feature Engineering
Extra features are created to improve model performance:
- Peak hour flag
- Weekend flag
- Rush hour indicator
- Weather impact score
- Traffic density score

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib, Seaborn
- Streamlit

## Project Structure
```text
Innovexa_Traffic_Demand_Prediction_Project/
├── app.py
├── train_model.py
├── visualizations.py
├── requirements.txt
├── README.md
├── data/
│   └── smart_city_traffic_sample.csv
├── models/
│   └── traffic_demand_model.pkl
├── reports/
│   └── model_metrics.txt
└── src/
    └── features.py
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Generate visualizations
```bash
python visualizations.py
```

### 4. Run Streamlit app
```bash
streamlit run app.py
```

## Output
The app predicts:
- Traffic demand in vehicles/hour
- Congestion level: Low, Medium, or High
- Peak traffic alert

## Future Enhancements
- Real-time traffic API integration
- Accident risk prediction
- Route recommendation system
- Traffic heatmap visualization
- Live dashboard for city traffic departments
