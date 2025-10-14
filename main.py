import pandas as pd
import data_preproccesing
from model import lstm as lstm
from sklearn.preprocessing import MinMaxScaler


def start_model():
    train, test, scaler = data_preproccesing.get_data(ratio = 0.7)
    lstm.run_model(train, test, scaler, epochs = 1, learning_rate= 0.001, write_to_file = False)
start_model()


