import pandas as pd
import data_preproccesing
from model import lstm, magnn
from sklearn.preprocessing import MinMaxScaler


def start_model():
    train, test, scaler = data_preproccesing.get_data()
    lstm.run_model(train, test, scaler, epochs = 1, )
start_model()
