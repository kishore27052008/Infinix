"""
risk_model.py
Combines volatility, beta, and avg_correlation into a single
0-100 risk score per company. Higher score = riskier.

Method: simple weighted sum (fully transparent, no black box) --
each raw feature is normalized to a 0-1 scale, multiplied by a
weight reflecting how much it contributes to risk, then summed
and rescaled to 0-100.
"""

import pandas as pd

# Weights must sum to 1.0 -- this makes the final score a clean
# weighted average, easy to justify to judges.
WEIGHTS = {
    "volatility": 0.5,        # biggest driver of risk -- price swings
    "beta": 0.3,               # market sensitivity
    "avg_correlation": 0.2,    # portfolio-level risk (less holding-level)
}


def normalize(series: pd.Series) -> pd.Series:
    """
    Min-max normalize a column to a 0-1 range:
    (value - min) / (max - min)
    Puts volatility, beta, and correlation on the same scale before
    combining them, since they have very different raw ranges.
    """
    return (series - series.min()) / (series.max() - series.min())


def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the feature DataFrame and returns it with new columns:
    - normalized versions of the three risk inputs (for transparency)
    - risk_score: final 0-100 combined score
    """
    df = df.copy()  # never mutate the caller's original DataFrame

    # Step 1: normalize each risk input to 0-1
    norm_volatility = normalize(df["volatility"])
    norm_beta = normalize(df["beta"])
    norm_correlation = normalize(df["avg_correlation"])

    # Step 2: weighted sum -> still 0-1 scale (since weights sum to 1
    # and each input is 0-1)
    weighted_score = (
        norm_volatility * WEIGHTS["volatility"]
        + norm_beta * WEIGHTS["beta"]
        + norm_correlation * WEIGHTS["avg_correlation"]
    )

    # Step 3: rescale 0-1 -> 0-100 for a more intuitive score
    df["risk_score"] = (weighted_score * 100).round(2)

    # Keep normalized components too -- useful later for explainability.py
    df["norm_volatility"] = norm_volatility.round(3)
    df["norm_beta"] = norm_beta.round(3)
    df["norm_correlation"] = norm_correlation.round(3)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/final_features.csv")
    df = df.rename(columns={"Unnamed: 0": "ticker"})   # <-- add this line

    scored = calculate_risk_score(df)

    print(scored[["ticker", "risk_score", "norm_volatility", "norm_beta", "norm_correlation"]]
          .sort_values("risk_score", ascending=False))