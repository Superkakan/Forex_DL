import yfinance as yf


def download_data(currency = "EURUSD=X", start = "2023-05-07", end = "2025-05-01", interval = "1h"): #EURUSD=X
    data = yf.download(tickers = currency, start=start, end=end, interval = interval)
    print("Saving to csv:")
    data.to_csv("yfinance_data/usdsek_yf.csv")

download_data()
    
