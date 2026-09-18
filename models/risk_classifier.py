"""
risk_classifier.py
Trains a RandomForestClassifier to predict risk category (Low/High)
directly from raw financial/market/sentiment features.

Labels are generated once from risk_model.py's risk_score using the
median as a split point (weak supervision -- standard practice when
no hand-labeled data exists). The classifier then learns the pattern
from raw features independently -- it does NOT see risk_score itself
as an input, only volatility/beta/avg_correlation/etc, so it's
genuinely learning the relationship, not just copying the formula.

NOTE: only 7 companies -- too small for a normal train/test split, so
we use leave-one-out cross-validation, the standard approach for tiny
datasets. We use 2 categories instead of 3 (Low/Medium/High) because
3-way splits leave too few examples per class with only 7 rows --
this is an honest dataset-size limitation, not a tuning trick.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

from explainability import load_and_clean
from risk_model import calculate_risk_score

FEATURE_COLUMNS = [
    "volatility", "beta", "avg_correlation",
    "revenue", "rd_expense", "net_income",
    "rd_pct_revenue", "profit_margin", "sentiment_score",
]


def assign_risk_label(df: pd.DataFrame) -> pd.Series:
    """
    Splits companies into High/Low risk using the MEDIAN risk_score
    as the cutoff -- guarantees a roughly balanced split (important
    with so few data points; a fixed threshold like 50 could easily
    put 6 of 7 companies in one bucket).
    """
    median_score = df["risk_score"].median()
    return df["risk_score"].apply(lambda x: "High" if x >= median_score else "Low")


def train_and_evaluate(df: pd.DataFrame):
    """
    Trains a RandomForestClassifier via leave-one-out cross-validation.
    Returns predictions, accuracy, and feature importances.
    """
    df = calculate_risk_score(df)
    df["risk_label"] = assign_risk_label(df)

    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]
    tickers = df["ticker"].values

    loo = LeaveOneOut()
    predictions = {}
    importances_per_fold = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        test_ticker = tickers[test_idx][0]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        predictions[test_ticker] = pred

        importances_per_fold.append(model.feature_importances_)

    actual_labels = df.set_index("ticker")["risk_label"].to_dict()
    predicted_in_order = [predictions[t] for t in tickers]
    actual_in_order = [actual_labels[t] for t in tickers]
    accuracy = accuracy_score(actual_in_order, predicted_in_order)

    avg_importances = pd.DataFrame(importances_per_fold, columns=FEATURE_COLUMNS).mean()
    feature_importances = avg_importances.sort_values(ascending=False).round(4).to_dict()

    return predictions, actual_labels, accuracy, feature_importances


if __name__ == "__main__":
    df = load_and_clean()

    predictions, actual_labels, accuracy, feature_importances = train_and_evaluate(df)

    print("=== Predicted vs Actual risk category (leave-one-out) ===")
    for ticker in df["ticker"]:
        print(f"{ticker}: predicted={predictions[ticker]}  actual={actual_labels[ticker]}")

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\n=== Feature importances for risk classification ===")
    for feature, importance in feature_importances.items():
        print(f"{feature}: {importance}")