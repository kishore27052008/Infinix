"""
return_predictor.py
Trains a RandomForestRegressor to predict avg_return from financial,
market, and sentiment features. This is a genuinely trained ML
component (risk_model.py is a transparent formula by design, for
explainability -- this file adds learned predictions and model-driven
feature importance).

NOTE: only 7 companies in this dataset -- far too small for a real
train/test split. We use leave-one-out cross-validation instead,
the standard approach for very small datasets. This is a
proof-of-concept of the ML pipeline, not a production-grade model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

from explainability import load_and_clean

FEATURE_COLUMNS = [
    "volatility", "beta", "avg_correlation",
    "revenue", "rd_expense", "net_income",
    "rd_pct_revenue", "profit_margin", "sentiment_score",
]
TARGET_COLUMN = "avg_return"


def train_and_evaluate(df: pd.DataFrame):
    """
    Trains a RandomForestRegressor using leave-one-out cross-validation
    (since we only have 7 rows -- a normal train/test split isn't
    meaningful at this size).
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    tickers = df["ticker"].values

    loo = LeaveOneOut()
    predictions = {}
    importances_per_fold = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        test_ticker = tickers[test_idx][0]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]
        predictions[test_ticker] = round(pred, 4)

        importances_per_fold.append(model.feature_importances_)

    actuals = y.values
    preds_in_order = [predictions[t] for t in tickers]
    mae = mean_absolute_error(actuals, preds_in_order)

    avg_importances = pd.DataFrame(importances_per_fold, columns=FEATURE_COLUMNS).mean()
    feature_importances = avg_importances.sort_values(ascending=False).round(4).to_dict()

    return predictions, mae, feature_importances


if __name__ == "__main__":
    df = load_and_clean()

    predictions, mae, feature_importances = train_and_evaluate(df)

    print("=== Predicted vs Actual avg_return (leave-one-out) ===")
    for ticker in df["ticker"]:
        actual = df.loc[df["ticker"] == ticker, "avg_return"].values[0]
        print(f"{ticker}: predicted={predictions[ticker]:.4f}  actual={actual:.4f}")

    print(f"\nMean Absolute Error: {mae:.4f}")

    print("\n=== Feature importances (what the model actually learned mattered) ===")
    for feature, importance in feature_importances.items():
        print(f"{feature}: {importance}")