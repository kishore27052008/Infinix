import requests
import pandas as pd
import time

# EDGAR requires a User-Agent header identifying who's making the request
HEADERS = {"User-Agent": "YourName YourEmail@example.com"}

CIK_MAP = {
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "AMD": "0000002488",
    "TSM": "0001046179",
    "AMZN": "0001018724",
}

def fetch_company_facts(cik):
    """Pulls the full set of reported financial facts for one company."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def extract_latest_value(facts, tag):
    """Pulls the most recent annual value for a given financial tag, e.g. 'Revenues'."""
    try:
        entries = facts["facts"]["us-gaap"][tag]["units"]["USD"]
        # Keep only annual (10-K) filings, sorted by end date, take the latest
        annual_entries = [e for e in entries if e.get("form") == "10-K"]
        if not annual_entries:
            return None
        latest = sorted(annual_entries, key=lambda x: x["end"])[-1]
        return latest["val"]
    except (KeyError, IndexError):
        return None

def build_financial_features():
    rows = []
    for ticker, cik in CIK_MAP.items():
        print(f"Fetching {ticker}...")
        facts = fetch_company_facts(cik)

        revenue = extract_latest_value(facts, "Revenues")
        rd_expense = extract_latest_value(facts, "ResearchAndDevelopmentExpense")
        net_income = extract_latest_value(facts, "NetIncomeLoss")

        rd_pct_revenue = (rd_expense / revenue) if revenue and rd_expense else None
        profit_margin = (net_income / revenue) if revenue and net_income else None

        rows.append({
            "ticker": ticker,
            "revenue": revenue,
            "rd_expense": rd_expense,
            "net_income": net_income,
            "rd_pct_revenue": rd_pct_revenue,
            "profit_margin": profit_margin,
        })

        time.sleep(0.5)  # be polite to SEC's servers, avoid rate limiting

    return pd.DataFrame(rows).set_index("ticker")

if __name__ == "__main__":
    features = build_financial_features()
    print(features)
    features.to_csv("data/processed/financial_features.csv")