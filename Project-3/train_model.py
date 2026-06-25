import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import add_features

DATA_PATH = "data/smart_city_traffic_sample.csv"
MODEL_PATH = "models/traffic_demand_model.pkl"
REPORT_PATH = "reports/model_metrics.txt"

def get_regressors():
    models = []
    models.append(("random_forest", RandomForestRegressor(n_estimators=200, random_state=42)))

    try:
        from xgboost import XGBRegressor
        models.append(("xgboost", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        )))
    except Exception:
        print("XGBoost not available. Continuing without it.")

    try:
        from lightgbm import LGBMRegressor
        models.append(("lightgbm", LGBMRegressor(
            n_estimators=300, learning_rate=0.05, random_state=42
        )))
    except Exception:
        print("LightGBM not available. Continuing without it.")

    if len(models) > 1:
        return VotingRegressor(estimators=models)
    return models[0][1]

def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    target = "traffic_demand"
    drop_cols = [target, "timestamp"]
    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", get_regressors())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    report = f"""Traffic Demand Prediction - Model Metrics
-----------------------------------------
R2 Score: {r2:.4f}
RMSE: {rmse:.2f}
MAE: {mae:.2f}
MAPE: {mape:.2f}%

Model saved at: {MODEL_PATH}
"""
    print(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()
