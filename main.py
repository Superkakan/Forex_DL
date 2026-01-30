import pandas as pd
import data_preproccesing
from model import lstm, lstm_keras
from sklearn.preprocessing import MinMaxScaler


def start_model():
    if (False): 
        train, test, scaler = data_preproccesing.get_data(ratio = 0.7)
    if (True):
        train, test, scaler = data_preproccesing.get_1m_forex_data(data_path="data/usdsek/DAT_XLSX_USDSEK_M1_2023.csv",ratio = 0.7)
    keras_benchmark = False
    pred_step = 10
    epochs = 10
    learning_rate= 0.001
    if (keras_benchmark == True):
        lstm_keras.run_benchmark(train,test,scaler)
    lstm.run_model(train, test, scaler, pred_step, epochs, learning_rate, write_to_file = False)
start_model()


