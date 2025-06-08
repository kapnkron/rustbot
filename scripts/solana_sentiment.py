import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Settings
query = "solana OR $SOL OR #solana"
max_tweets = 500

# Define output path
output_dir = os.path.expanduser("~/bot_project/Beatrice/data/processed")
os.makedirs(output_dir, exist_ok=True)

# Scrape tweets
tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= max_tweets:
        break
    tweets.append([tweet.date, tweet.content, tweet.user.username])

# Convert to DataFrame
df = pd.DataFrame(tweets, columns=["date", "content", "user"])

# Sentiment Analysis
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df["sentiment"] = df["content"].apply(get_sentiment)

# Group by day for trend
df["date_only"] = pd.to_datetime(df["date"]).dt.date
daily_sentiment = df.groupby("date_only")["sentiment"].mean()

# Save results
csv_path = os.path.join(output_dir, "solana_tweets_sentiment.csv")
plot_path = os.path.join(output_dir, "solana_sentiment_plot.png")

df.to_csv(csv_path, index=False)
print(f"Saved CSV to: {csv_path}")

# Plot and save sentiment chart
plt.figure(figsize=(10, 5))
daily_sentiment.plot(marker='o')
plt.title("Solana Sentiment Over Time (Twitter)")
plt.xlabel("Date")
plt.ylabel("Avg Sentiment")
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
print(f"Saved plot to: {plot_path}")

