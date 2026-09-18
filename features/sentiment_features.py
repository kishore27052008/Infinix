from transformers import pipeline
import pandas as pd

# Load FinBERT once — a model specifically trained on financial text sentiment
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Small set of illustrative headlines per company for now.
# Can be swapped for real scraped headlines later if time allows.
HEADLINES = {
    "NVDA": [
        "NVIDIA reports record quarterly revenue driven by AI chip demand",
        "Analysts raise concerns over NVIDIA's high valuation amid AI hype",
    ],
    "MSFT": [
        "Microsoft's cloud and AI investments boost quarterly earnings",
        "Microsoft faces regulatory scrutiny over AI partnerships",
    ],
    "GOOGL": [
        "Google's Gemini AI model shows strong performance in benchmarks",
        "Alphabet stock dips on concerns over AI search competition",
    ],
    "META": [
        "Meta's AI-driven ad targeting improves revenue per user",
        "Meta faces criticism over AI content moderation failures",
    ],
    "AMD": [
        "AMD's new AI chips gain traction against NVIDIA in data centers",
        "AMD stock volatile amid supply chain concerns",
    ],
    "AVGO": [
        "Broadcom's AI chip revenue surges on hyperscaler demand",
        "Broadcom stock faces pressure from broader semiconductor slowdown",
    ],
    "AMZN": [
        "Amazon Web Services sees strong growth from AI infrastructure demand",
        "Amazon's AI investments weigh on short-term profit margins",
    ],
}

def compute_sentiment_features():
    rows = []
    for ticker, headlines in HEADLINES.items():
        scores = []
        for headline in headlines:
            result = finbert(headline)[0]
            if result["label"] == "positive":
                scores.append(result["score"])
            elif result["label"] == "negative":
                scores.append(-result["score"])
            else:
                scores.append(0)

        avg_sentiment = sum(scores) / len(scores)
        rows.append({"ticker": ticker, "sentiment_score": avg_sentiment})

    return pd.DataFrame(rows).set_index("ticker")

if __name__ == "__main__":
    features = compute_sentiment_features()
    print(features)
    features.to_csv("data/processed/sentiment_features.csv")