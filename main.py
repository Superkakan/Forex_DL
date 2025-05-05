import pandas as pd
import data_preproccesing
from model import lstm, magnn
from sklearn.preprocessing import MinMaxScaler


def start_model():
    train, test, scaler = data_preproccesing.get_data(ratio = 0.9)
    lstm.run_model(train, test, scaler, epochs = 340, write_to_file = False)
start_model()
