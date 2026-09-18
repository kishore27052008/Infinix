import pandas as pd

def build_all_features():
    market = pd.read_csv("data/processed/market_features.csv", index_col=0)
    financial = pd.read_csv("data/processed/financial_features.csv", index_col=0)
    sentiment = pd.read_csv("data/processed/sentiment_features.csv", index_col=0)

    # market_features.csv is indexed by ticker already (e.g. NVDA, MSFT...)
    # financial_features.csv and sentiment_features.csv use the same tickers
    combined = market.join(financial, how="outer").join(sentiment, how="outer")

    return combined

if __name__ == "__main__":
    final_table = build_all_features()
    print(final_table)
    final_table.to_csv("data/processed/final_features.csv")