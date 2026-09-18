"""
explainability.py
Breaks down which features drove each company's risk score and
overall attractiveness. Not true SHAP values -- just transparent
attribution based on the same weights used in risk_model.py and
a parallel weighting for "attractiveness" factors. Output is
structured so it can feed a bar chart directly (feature -> contribution).
"""

import pandas as pd
from risk_model import WEIGHTS, calculate_risk_score, normalize

# Weights for "attractiveness" -- separate from risk weights.
# These are the factors that make a company a GOOD investment,
# independent of how risky it is. Must sum to 1.0.
# Note: revenue_growth and sentiment_trend aren't in the real dataset,
# so weight is redistributed across what IS available.
ATTRACTIVENESS_WEIGHTS = {
    "profit_margin": 0.30,
    "rd_pct_revenue": 0.30,
    "sentiment_score": 0.25,
    "avg_return": 0.15,
}


def load_and_clean(path: str = "data/processed/final_features.csv") -> pd.DataFrame:
    """
    Loads the real feature table, fixes the ticker column name, and
    fills any missing values with the column average -- some companies
    (e.g. AMD, AMZN) are missing rd_pct_revenue/profit_margin because
    the underlying SEC data wasn't cleanly available for them.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "ticker"})

    for col in ["rd_pct_revenue", "profit_margin"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    return df


def explain_risk(df: pd.DataFrame) -> dict:
    """
    Returns, for each company, how many risk_score points each of the
    three risk features contributed. Points sum to that company's risk_score.
    """
    scored = calculate_risk_score(df)
    explanations = {}

    for _, row in scored.iterrows():
        explanations[row["ticker"]] = {
            "volatility": round(row["norm_volatility"] * WEIGHTS["volatility"] * 100, 2),
            "beta": round(row["norm_beta"] * WEIGHTS["beta"] * 100, 2),
            "avg_correlation": round(row["norm_correlation"] * WEIGHTS["avg_correlation"] * 100, 2),
        }

    return explanations


def explain_attractiveness(df: pd.DataFrame) -> dict:
    """
    Same idea as explain_risk, but for "attractiveness" factors --
    profit margin, R&D investment, sentiment, and returns.
    """
    df = df.copy()

    norm_margin = normalize(df["profit_margin"])
    norm_rd = normalize(df["rd_pct_revenue"])
    norm_sentiment = normalize(df["sentiment_score"])
    norm_return = normalize(df["avg_return"])

    explanations = {}
    for i, ticker in enumerate(df["ticker"]):
        explanations[ticker] = {
            "profit_margin": round(norm_margin.iloc[i] * ATTRACTIVENESS_WEIGHTS["profit_margin"] * 100, 2),
            "rd_pct_revenue": round(norm_rd.iloc[i] * ATTRACTIVENESS_WEIGHTS["rd_pct_revenue"] * 100, 2),
            "sentiment_score": round(norm_sentiment.iloc[i] * ATTRACTIVENESS_WEIGHTS["sentiment_score"] * 100, 2),
            "avg_return": round(norm_return.iloc[i] * ATTRACTIVENESS_WEIGHTS["avg_return"] * 100, 2),
        }

    return explanations


def summarize_company(ticker: str, risk_expl: dict, attract_expl: dict) -> str:
    """
    Turns the raw numbers into a short human-readable sentence, e.g.:
    "NVDA: high rd_pct_revenue (+), high volatility (-)"
    Picks the single strongest risk factor and strongest attractiveness
    factor for a punchy one-liner -- useful for a demo/dashboard.
    """
    top_risk_factor = max(risk_expl[ticker], key=risk_expl[ticker].get)
    top_attract_factor = max(attract_expl[ticker], key=attract_expl[ticker].get)

    labels = {
        "volatility": "high volatility",
        "beta": "high market sensitivity",
        "avg_correlation": "high correlation to peers",
        "profit_margin": "strong profit margin",
        "rd_pct_revenue": "high R&D investment",
        "sentiment_score": "positive sentiment",
        "avg_return": "strong average return",
    }

    return f"{ticker}: {labels[top_attract_factor]} (+), {labels[top_risk_factor]} (-)"


if __name__ == "__main__":
    df = load_and_clean()

    risk_expl = explain_risk(df)
    attract_expl = explain_attractiveness(df)

    print("=== Risk factor breakdown (points out of each company's risk_score) ===")
    for ticker, factors in risk_expl.items():
        print(f"{ticker}: {factors}")

    print("\n=== Attractiveness factor breakdown ===")
    for ticker, factors in attract_expl.items():
        print(f"{ticker}: {factors}")

    print("\n=== One-line summaries ===")
    for ticker in df["ticker"]:
        print(summarize_company(ticker, risk_expl, attract_expl))