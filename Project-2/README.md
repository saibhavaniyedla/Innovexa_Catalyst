# Credit Risk Default Prediction

This ZIP contains both sample CSV files and full ML code.

## Project Meaning

This project predicts whether a loan applicant may default in the next 12 months.

Target column:

```text
default_12m
1 = will default
0 = will not default
```

## Included CSV Files

Inside `data/`:

- `application.csv`
- `bureau.csv`
- `previous_loans.csv`
- `payments.csv`
- `credit_card.csv`
- `train_labels.csv`
- `test.csv`

These are synthetic sample datasets created for practice and project submission demo.

## How to Run

Open terminal inside this folder and run:

```bash
pip install -r requirements.txt
python src/train_model.py
```

## Output Files

After running the code, files will be created in `outputs/`:

- `model_results.csv`
- `credit_risk_predictions.csv`

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Metrics Used

- AUC-ROC
- F1 Score
- Precision
- Recall
- PR-AUC
- KS Statistic

## Note

The CSV files are sample generated datasets. For final internship/hackathon submission, replace these files with the official dataset if they provide one.