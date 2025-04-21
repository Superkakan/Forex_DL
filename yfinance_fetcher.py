import yfinance as yf


def download_data(currency = "EURUSD=X", start = "2023-04-23", end = "2025-04-21", interval = "1h"):
    data = yf.download(tickers = currency, start=start, end=end, interval = interval)
    print("Saving to csv:")
    data.to_csv("yfinance_data/eurusd_yf.csv")
    
