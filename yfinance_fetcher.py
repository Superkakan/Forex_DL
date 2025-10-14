import yfinance as yf



def download_data(currency = "EURUSD=X", interval = "1d"): #EURUSD=X start = "2018-05-07", end = "2025-05-01", 
    #test yfinance first
    data = yf.download(tickers = currency, interval = interval) # start=start, end=end,
    if (data.empty): # if yfinance not working try finnhub
        pass #finnhub
    else:
        pass
    
    if (not data.empty):
        print("Saving to csv:")
        data.to_csv("data/yfinance_data/eurusd_yf_1d.csv")
    else:
        print("Data could not be downloaded")


#download_data()

    
