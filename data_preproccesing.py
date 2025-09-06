import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance_fetcher
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def split_data(data, ratio=0.7):
    split_index = int(len(data) * ratio)
    train, test = data[:split_index], data[split_index:]
    return train, test

def get_data(ratio=0.7):
    df = pd.read_csv("Forex_DL/yfinance_data/eurusd_1d.csv")  # skip the Ticker row
    df = df.rename(columns={"Date": "Datetime"})  # Rename 'Price' to 'Datetime'
    df = df.rename(columns={"Close/Last" : "Close"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Date"] = df["Datetime"].dt.date  # extract date part for sentiment merge
    df = df.drop(columns="Volume")

    sentiment = news_data_preproceessing()
    sentiment["Date"] = pd.to_datetime(sentiment["Date"]).dt.date

    df = pd.merge(df, sentiment, on="Date", how="left")
    df["TotalSentiment"] = df["TotalSentiment"].fillna(0)

    df = df[["Close", "TotalSentiment"]]
    df = df.astype(float)
    # Normalize
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    train, test = split_data(df_scaled, ratio)
    print("Train shape:", train.shape, "Test shape:", test.shape)
    return train, test, scaler

    

#Ecb positive == good

def news_data_preproceessing():
    # Load datasets
    ecb_data = pd.read_csv("Forex_DL/news_data/ecb_classified.csv")
    fed_article_data = pd.read_csv("Forex_DL/news_data/fed_articles_classified.csv")
    fed_speeches_data = pd.read_csv("Forex_DL/news_data/fed_speeches_classified.csv")

    # Remove unnecessary columns
    for df in [ecb_data, fed_article_data, fed_speeches_data]:
        df.drop(columns=["Title", "Reasoning"], inplace=True)

    # Sentiment maps
    sentiment_map_eur = {
        "Very Negative": -0.02,
        "Negative": -0.01,
        "Neutral": 0.00,
        "Positive": 0.01,
        "Very Positive": 0.02
    }
    sentiment_map_usd = {
        "Very Negative": 0.02,
        "Negative": 0.01,
        "Neutral": 0.00,
        "Positive": -0.01,
        "Very Positive": -0.02
    }

    # Clean and map ECB (EUR-positive)
    ecb_data["Classification"] = ecb_data["Classification"].str.replace("<", "").str.replace(">", "").str.strip()
    ecb_data["SentimentScore"] = ecb_data["Classification"].map(sentiment_map_eur)
    ecb_data["Date"] = pd.to_datetime(ecb_data["Date"])
    daily_ecb_sentiment = ecb_data.groupby("Date")["SentimentScore"].sum().reset_index()

    # Clean and map Fed data (USD-positive = EUR/USD-negative)
    for df in [fed_article_data, fed_speeches_data]:
        df["Classification"] = df["Classification"].str.replace("<", "").str.replace(">", "").str.strip()
        df["SentimentScore"] = df["Classification"].map(sentiment_map_usd)
        df["Date"] = pd.to_datetime(df["Date"])

    daily_fed_article_sentiment = fed_article_data.groupby("Date")["SentimentScore"].sum()
    daily_fed_speech_sentiment = fed_speeches_data.groupby("Date")["SentimentScore"].sum()

    # Merge all USD sources
    daily_fed_sentiment = pd.concat([daily_fed_article_sentiment, daily_fed_speech_sentiment], axis=0)
    daily_fed_sentiment = daily_fed_sentiment.groupby("Date").sum().reset_index()

    # Combine ECB and Fed
    sentiment_all = pd.merge(daily_ecb_sentiment, daily_fed_sentiment, on="Date", how="outer", suffixes=("_ECB", "_Fed"))
    sentiment_all.fillna(0, inplace=True)
    sentiment_all["TotalSentiment"] = sentiment_all["SentimentScore_ECB"] + sentiment_all["SentimentScore_Fed"]
    
    # Sort by date
    sentiment_all.sort_values("Date", inplace=True)
    sentiment_all = sentiment_all.drop(columns=["SentimentScore_ECB", "SentimentScore_Fed"])
    print(sentiment_all.head(10))

    return sentiment_all
