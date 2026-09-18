import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["NVDA", "MSFT", "GOOGL", "META", "AMD", "AVGO", "AMZN"]

def fetch_price_data(tickers, start="2023-01-01", end="2025-09-18"):
    """Downloads daily price history for all tickers."""
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data

def compute_returns(price_df):
    """Daily % change in price for each company."""
    return price_df.pct_change().dropna()

def compute_volatility(returns_df):
    """Standard deviation of daily returns, annualized. Higher = riskier."""
    daily_vol = returns_df.std()
    annualized_vol = daily_vol * np.sqrt(252)
    return annualized_vol

def compute_avg_return(returns_df):
    """Average daily return, annualized."""
    daily_avg = returns_df.mean()
    annualized_return = daily_avg * 252
    return annualized_return

def compute_correlation_matrix(returns_df):
    """How much each company's returns move together. Key for diversification."""
    return returns_df.corr()

def compute_beta(returns_df, market_ticker="^GSPC", start="2023-01-01", end="2025-09-18"):
    """Beta = how volatile a stock is relative to the overall market (S&P 500)."""
    market_data = yf.download(market_ticker, start=start, end=end)["Close"]
    market_returns = market_data.pct_change().dropna()

    betas = {}
    for company in returns_df.columns:
        aligned = pd.concat([returns_df[company], market_returns], axis=1).dropna()
        aligned.columns = ["company", "market"]
        covariance = aligned["company"].cov(aligned["market"])
        market_variance = aligned["market"].var()
        betas[company] = covariance / market_variance
    return pd.Series(betas)

def build_market_features():
    """Main function: runs everything and returns one clean feature table."""
    prices = fetch_price_data(TICKERS)
    returns = compute_returns(prices)

    features = pd.DataFrame({
        "avg_return": compute_avg_return(returns),
        "volatility": compute_volatility(returns),
        "beta": compute_beta(returns),
    })

    corr_matrix = compute_correlation_matrix(returns)
    features["avg_correlation"] = corr_matrix.apply(
        lambda row: row.drop(row.name).mean(), axis=1
    )

    return features, corr_matrix

if __name__ == "__main__":
    features, corr_matrix = build_market_features()
    print(features)
    features.to_csv("data/processed/market_features.csv")
    corr_matrix.to_csv("data/processed/correlation_matrix.csv")