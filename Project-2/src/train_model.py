import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from utils import safe_ratio, evaluate_model


DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    application = pd.read_csv(os.path.join(DATA_DIR, "application.csv"))
    bureau = pd.read_csv(os.path.join(DATA_DIR, "bureau.csv"))
    previous_loans = pd.read_csv(os.path.join(DATA_DIR, "previous_loans.csv"))
    payments = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
    credit_card = pd.read_csv(os.path.join(DATA_DIR, "credit_card.csv"))
    train_labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    return application, bureau, previous_loans, payments, credit_card, train_labels, test


def create_features(application, bureau, previous_loans, payments, credit_card, train_labels=None):
    bureau_agg = bureau.groupby("id").agg({
        "bureau_score": ["mean", "min", "max"],
        "num_of_accounts": "sum",
        "num_of_open_accounts": "sum",
        "total_debt": "sum",
        "delinquent_accounts": "sum"
    }).reset_index()

    bureau_agg.columns = [
        "id", "bureau_score_mean", "bureau_score_min", "bureau_score_max",
        "total_accounts", "open_accounts", "total_bureau_debt", "delinquent_accounts"
    ]

    previous_agg = previous_loans.groupby("id").agg({
        "num_loans": "sum",
        "num_defaults": "sum",
        "total_loan_amount": "sum",
        "total_repaid_amount": "sum",
        "avg_dpd": "mean"
    }).reset_index()

    previous_agg["repayment_ratio"] = safe_ratio(
        previous_agg["total_repaid_amount"],
        previous_agg["total_loan_amount"]
    )

    payments_agg = payments.groupby("id").agg({
        "months_on_book": "max",
        "dpd": ["mean", "max", "sum"],
        "payment_amount": ["mean", "sum"]
    }).reset_index()

    payments_agg.columns = [
        "id", "months_on_book", "dpd_mean", "dpd_max", "dpd_sum",
        "payment_amount_mean", "payment_amount_sum"
    ]

    credit_agg = credit_card.groupby("id").agg({
        "num_cards": "sum",
        "credit_limit": "sum",
        "utilization_ratio": "mean",
        "max_utilization_ratio": "max",
        "late_payments": "sum"
    }).reset_index()

    data = application.copy()
    data = data.merge(bureau_agg, on="id", how="left")
    data = data.merge(previous_agg, on="id", how="left")
    data = data.merge(payments_agg, on="id", how="left")
    data = data.merge(credit_agg, on="id", how="left")

    data["debt_to_income_ratio"] = safe_ratio(data["total_bureau_debt"], data["annual_income"])
    data["loan_to_income_ratio"] = safe_ratio(data["loan_amount"], data["annual_income"])
    data["credit_utilization_risk"] = data["utilization_ratio"] * data["credit_limit"]
    data["default_history_ratio"] = safe_ratio(data["num_defaults"], data["num_loans"])

    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    if train_labels is not None:
        data = data.merge(train_labels, on="id", how="left")

    return data


def encode_categorical(train_df, test_df):
    cat_cols = train_df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined)

        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    return train_df, test_df


def main():
    print("Loading CSV files...")
    application, bureau, previous_loans, payments, credit_card, train_labels, test = load_data()

    print("Building train features...")
    train_data = create_features(application, bureau, previous_loans, payments, credit_card, train_labels)

    print("Building test features...")
    test_data = create_features(test, bureau, previous_loans, payments, credit_card)

    y = train_data["default_12m"]
    X = train_data.drop(columns=["id", "default_12m"])
    test_ids = test_data["id"]
    test_features = test_data.drop(columns=["id"])

    X, test_features = encode_categorical(X, test_features)
    test_features = test_features.reindex(columns=X.columns, fill_value=0)

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    test_features = pd.DataFrame(imputer.transform(test_features), columns=X.columns)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = []
    best_model = None
    best_auc = -1
    best_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_smote, y_train_smote)

        y_pred = model.predict(X_valid)
        y_prob = model.predict_proba(X_valid)[:, 1]

        metrics = evaluate_model(name, y_valid, y_pred, y_prob)
        results.append(metrics)

        if metrics["AUC_ROC"] > best_auc:
            best_auc = metrics["AUC_ROC"]
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)

    print("\nModel Results:")
    print(results_df)

    print(f"\nBest Model: {best_name}")

    y_pred_best = best_model.predict(X_valid)
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred_best))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, y_pred_best))

    test_prob = best_model.predict_proba(test_features)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    submission = pd.DataFrame({
        "id": test_ids,
        "default_probability": test_prob,
        "predicted_default": test_pred
    })

    output_file = os.path.join(OUTPUT_DIR, "credit_risk_predictions.csv")
    submission.to_csv(output_file, index=False)

    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    main()