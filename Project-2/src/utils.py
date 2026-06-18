import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score


def safe_ratio(numerator, denominator):
    return numerator / (denominator + 1)


def ks_statistic(y_true, y_prob):
    data_ks = pd.DataFrame({
        "target": y_true,
        "probability": y_prob
    }).sort_values("probability", ascending=False)

    bad_total = max((data_ks["target"] == 1).sum(), 1)
    good_total = max((data_ks["target"] == 0).sum(), 1)

    data_ks["cum_bad"] = (data_ks["target"] == 1).cumsum() / bad_total
    data_ks["cum_good"] = (data_ks["target"] == 0).cumsum() / good_total
    data_ks["ks"] = abs(data_ks["cum_bad"] - data_ks["cum_good"])

    return data_ks["ks"].max()


def evaluate_model(model_name, y_true, y_pred, y_prob):
    return {
        "Model": model_name,
        "AUC_ROC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "KS_Statistic": ks_statistic(y_true, y_prob)
    }