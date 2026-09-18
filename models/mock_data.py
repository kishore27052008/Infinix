"""
mock_data.py
Generates fake company feature data so we can build/test the ML layer
before the real feature table (from features/ scripts) is ready.
"""

import pandas as pd
import numpy as np

def generate_mock_data(seed: int = 42) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per company, matching the agreed
    feature table shape:
    company | revenue_growth | rd_pct_revenue | profit_margin |
    debt_to_equity | volatility | avg_return | beta |
    correlation_to_others | sentiment_score | sentiment_trend
    """
    np.random.seed(seed)  # keeps results reproducible every run

    companies = ["NVDA", "MSFT", "GOOGL", "META", "AMD", "TSM"]

    data = {
        "company": companies,

        # % YoY revenue growth. AI-heavy names skew high (NVDA, AMD).
        "revenue_growth": [0.62, 0.16, 0.14, 0.22, 0.35, 0.28],

        # R&D as % of revenue. Chip/software companies spend heavily.
        "rd_pct_revenue": [0.18, 0.13, 0.15, 0.21, 0.24, 0.08],

        # Net profit margin.
        "profit_margin": [0.55, 0.36, 0.29, 0.34, 0.12, 0.38],

        # Debt-to-equity ratio. Higher = more leveraged/risky.
        "debt_to_equity": [0.35, 0.42, 0.10, 0.31, 0.45, 0.28],

        # Annualized volatility (std dev of returns). Higher = riskier.
        "volatility": [0.55, 0.28, 0.30, 0.38, 0.60, 0.33],

        # Average historical return (annualized).
        "avg_return": [0.45, 0.22, 0.18, 0.25, 0.30, 0.20],

        # Beta: sensitivity to overall market moves. >1 = more volatile than market.
        "beta": [1.75, 0.95, 1.05, 1.30, 1.85, 1.15],

        # Average correlation to the *other* companies in the basket (0-1).
        # Used later to penalize putting money into similar-moving stocks.
        "correlation_to_others": [0.65, 0.55, 0.60, 0.58, 0.70, 0.50],

        # Sentiment score from news/social analysis, -1 (very negative) to +1 (very positive).
        "sentiment_score": [0.72, 0.40, 0.35, 0.20, 0.30, 0.45],

        # Sentiment trend: is sentiment improving or declining recently? -1 to +1.
        "sentiment_trend": [0.15, 0.05, -0.05, -0.10, 0.20, 0.08],
    }

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_mock_data()
    print(df)
    df.to_csv("data/raw/mock_company_features.csv", index=False)
    print("\nSaved to data/raw/mock_company_features.csv")