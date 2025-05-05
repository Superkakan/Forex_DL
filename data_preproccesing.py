import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance_fetcher
import os

def split_data(data, ratio = 0.7):
    split_index = int(len(data)*ratio)
    train, test = data[:split_index],data[split_index:]
    return train, test

def get_data(ratio = 0.7):
    if (not os.path.isfile("yfinance_data/eurusd_yf.csv")): #check if data exists, if not then download it
        yfinance_fetcher.download_data()

    dataframe = pd.read_csv("yfinance_data/eurusd_yf.csv")

    #dataframe = dataframe.round(decimals=5)
    dataframe = dataframe.drop(index=0) #Ticker
    dataframe = dataframe.drop(index=1) #Datetime
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe[["Close"]]
    dataframe = dataframe.dropna()
    #Convert to numpy array
    dataframe = dataframe.to_numpy()

    scaler = MinMaxScaler()
    dataframe = scaler.fit_transform(dataframe) 

    print(dataframe[:3])
    train, test = split_data(dataframe, ratio) #train_test_split(dataframe, train_size=0.7)
    print("Size of train and test: ", train.size, test.size)
    return train, test, scaler
    