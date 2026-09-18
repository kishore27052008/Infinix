"""
portfolio_builder.py
Suggests portfolio weights across companies using risk scores and
returns, favoring diversification. Method: simple, explainable
scoring (return-per-risk, penalized by correlation) followed by
normalization and a max-weight cap -- no optimization solver,
no black box.
"""

import pandas as pd
from risk_model import calculate_risk_score
from explainability import load_and_clean

MAX_WEIGHT = 0.30   # no single company can exceed 30% of the portfolio


def build_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the feature DataFrame (must include avg_return, avg_correlation)
    and returns it with a 'weight' column -- suggested portfolio allocation
    per company, as a percentage (0-1 scale, sums to 1.0).
    """
    df = calculate_risk_score(df)   # adds risk_score column
    df = df.copy()

    # Step A: reward-per-risk score. Small constant (1) added to risk_score
    # to avoid divide-by-zero if any company somehow scored 0 risk.
    df["raw_score"] = df["avg_return"] / (df["risk_score"] + 1)

    # Step B: penalize high correlation -- a company that moves in lockstep
    # with the rest of the basket adds less diversification value, even if
    # its raw return/risk looks good on its own.
    df["adjusted_score"] = df["raw_score"] * (1 - df["avg_correlation"])

    # Guard against negative scores (possible if avg_return is negative) --
    # can't have negative portfolio weight. Floor at a tiny positive number
    # instead of zero, so every company still gets some minimal weight
    # rather than being completely excluded.
    df["adjusted_score"] = df["adjusted_score"].clip(lower=0.001)

    # Step C: normalize into weights that sum to 1.0 (100%)
    total_score = df["adjusted_score"].sum()
    df["weight"] = df["adjusted_score"] / total_score

    # Step D: cap any company above MAX_WEIGHT, redistribute the excess
    # proportionally among the remaining (uncapped) companies.
    df = _apply_weight_cap(df)

    return df


def _apply_weight_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repeatedly clips any weight above MAX_WEIGHT down to the cap, and
    redistributes the removed excess proportionally across the companies
    still under the cap. Loops in case redistribution pushes another
    company over the cap too.
    """
    df = df.copy()

    for _ in range(10):  # safety limit on redistribution passes
        over_cap = df["weight"] > MAX_WEIGHT
        if not over_cap.any():
            break

        excess = (df.loc[over_cap, "weight"] - MAX_WEIGHT).sum()
        df.loc[over_cap, "weight"] = MAX_WEIGHT

        under_cap = ~over_cap
        under_cap_total = df.loc[under_cap, "weight"].sum()

        if under_cap_total > 0:
            df.loc[under_cap, "weight"] += (
                df.loc[under_cap, "weight"] / under_cap_total * excess
            )

    return df


if __name__ == "__main__":
    df = load_and_clean()
    result = build_portfolio(df)

    result["weight_pct"] = (result["weight"] * 100).round(2)

    print(result[["ticker", "risk_score", "avg_return", "avg_correlation", "weight_pct"]]
          .sort_values("weight_pct", ascending=False))

    print(f"\nTotal allocation: {result['weight_pct'].sum():.2f}%")