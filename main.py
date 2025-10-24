import pandas as pd
import data_preproccesing
from model import lstm as lstm
from sklearn.preprocessing import MinMaxScaler


def start_model():
    train, test, scaler = data_preproccesing.get_data(ratio = 0.7)
    lstm.run_model(train, test, scaler, pred_step = 1, epochs = 10, learning_rate= 0.01, write_to_file = False)
start_model()


