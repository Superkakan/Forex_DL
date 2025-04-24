import pandas as pd
from sklearn.model_selection import train_test_split
import yfinance_fetcher
import os

def split_data(data, ratio = 0.7):
    split_index = int(len(data)*ratio)
    train, test = data[:split_index],data[split_index:]
    return train, test

def get_data():
    if (not os.path.isfile("yfinance_data/eurusd_yf.csv")): #check if data exists, if not then download it
        yfinance_fetcher.download_data()

    dataframe = pd.read_csv("yfinance_data/eurusd_yf.csv")
    dataframe = dataframe.drop(index=0) #Ticker
    dataframe = dataframe.drop(index=1) #Datetime
    dataframe = dataframe.drop(columns = "Volume") #always 0
    print(dataframe[:3])
    train, test = split_data(dataframe) #train_test_split(dataframe, train_size=0.7)
    print("Size of train and test: ", train.size, test.size)

    
get_data()