#import tensorflow as tf
#from tensorflow import keras
#from keras import layers, Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import sys

from presentation.graph import graphing 

#def build_lstm_model(input_shape):
#    model = Sequential([
#        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
#        layers.Dropout(0.2),
#        layers.LSTM(32),
#        layers.Dense(1)
#    ])
#    model.compile(optimizer="adam", loss="mse")
#    return model

def sigmoid(x):
    result = (1 / (1 + np.exp(-x)))
    return result

def d_sigmoid(x):
    result = (sigmoid(x) * (1 - sigmoid(x)))
    return result

def tanh(x):
    result = ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) 
    return result

def d_tanh(x):
    result = (1 - np.pow(tanh(x)))
    return result


class LSTMCell:

    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        limit = np.sqrt(6 / (input_dim + hidden_dim))
        self.Wf = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        self.Wi = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        self.Wo = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        self.Wc = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))

        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))

    def forward(self, xt, h_prev, c_prev):
        concat = np.vstack((h_prev, xt)) #Shape hidden dim + input dim, 1

        #gates
        forget_gate = sigmoid(np.dot(self.Wf, concat) + self.bf) #Weight Forward, 
        input_gate = sigmoid(np.dot(self.Wi, concat) + self.bi)
        output_gate = sigmoid(np.dot(self.Wo, concat) + self.bo)
        candidate_cell_gate = tanh(np.dot(self.Wc, concat) + self.bc)
        new_cell_state = (forget_gate * c_prev) + (input_gate * candidate_cell_gate)
        hidden_state = output_gate * tanh(new_cell_state)

        self.cache = (  concat,
                        forget_gate,
                        input_gate,
                        output_gate,
                        candidate_cell_gate,
                        new_cell_state,
                        c_prev,
                        hidden_state,
                        xt)
        return hidden_state, new_cell_state


def generate_testdata(seq_len = 50, total_points = 1000): #sine wave data
    x = np.linspace(0,20 * np.pi, total_points)
    data = np.sin(x)
    sequences = []
    labels = []
    for i in range(len(data)- seq_len):
        seq = data[i:i+seq_len].reshape(-1,1,1)
        target = data[i+seq_len]
        sequences.append(seq)
        labels.append(target)
    return np.array(sequences), np.array(labels)


def create_sequences(data, seq_length = 32):
    X,y = [],[]
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        label = data[i + seq_length][0]
        X.append(sequence.reshape(seq_length, 2, 1))
        y.append(label)
        #X.append(data[i:i+seq_length].reshape(-1,2,1))
        #y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def predict_future(model, W_out, b_out, initial_seq, scaler, steps=24):
    lstm = model
    hidden_dim = lstm.hidden_dim
    predictions = []

    h = np.zeros((hidden_dim, 1))
    c = np.zeros((hidden_dim, 1))

    current_seq = initial_seq.copy()  # (seq_len, 1, num_features)
    num_features = current_seq.shape[2]

    for _ in range(steps):
        for t in range(len(current_seq)):
            h, c = lstm.forward(current_seq[t], h, c)

        y_pred = np.dot(W_out, h) + b_out  # Shape: (1, 1)
        pred_val = y_pred.item()
        predictions.append(pred_val)

        # Extract the last timestep (shape: (1, num_features))
        last_step = current_seq[-1]  # shape is (1, num_features)

        # Overwrite the predicted value (assuming first feature is the one being predicted)
        new_step = last_step.copy()
        new_step[0, 0] = pred_val #y_pred


        # Reshape to (1, 1, num_features) for concatenation
        new_input = new_step.reshape(1, 2, 1)

        # Append new input and remove oldest
        current_seq = np.concatenate([current_seq[1:], new_input], axis=0)

    # Inverse transform the predictions
    #predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 2))
    dummy_second_feature = np.tile(current_seq[-1][1, 0], (steps, 1))  # keep last second feature
    combined_preds = np.hstack((np.array(predictions).reshape(-1, 1), dummy_second_feature))

    predictions_inverse = scaler.inverse_transform(combined_preds)
    
    return predictions_inverse




def run_model(train_data, val_data, scaler, epochs = 5, learning_rate = 0.01, write_to_file = False):
    hidden_dim = 10
    input_dim = 2
    seq_len = 25
    
    if (write_to_file == True):
        sys.stdout = open("results.txt", "a")
        print("")

    lstm = LSTMCell(input_dim, hidden_dim)
    W_out = np.random.randn(1, hidden_dim) * 0.1 #Output layer weights
    b_out = np.zeros((1,1))

    X,y = create_sequences(train_data, seq_len) # Training data
    X_val, y_val = create_sequences(val_data, seq_len)
    for epoch in range(epochs):
        loss_epoch = 0
        for i in range(len(X)):
            x_seq = X[i]
            # h = hidden, c = candidate
            h = np.zeros((hidden_dim,1))
            c = np.zeros((hidden_dim,1))

            for t in range(seq_len):
                h,c = lstm.forward(x_seq[t],h,c)
            
            y_pred = np.dot(W_out, h) + b_out
            target = np.array([[y[i]]])
            error = y_pred - target #y[i]
            loss = (error ** 2).sum()
            loss_epoch += loss

            dW_out = 2 * error * h.T
            db_out = 2 * error

            W_out -= learning_rate * dW_out
            b_out -= learning_rate * db_out
        print(f"Epoch {epoch+1}: Loss: {loss_epoch / len(X):.4f}")
    #validation part
    preds = []
    targets = []
    for i in range(len(X_val)):
        x_seq = X_val[i]
        h = np.zeros((hidden_dim, 1))
        c = np.zeros((hidden_dim, 1))
        for j in range(seq_len):
            h, c = lstm.forward(x_seq[j], h, c)
        y_pred = np.dot(W_out, h) + b_out
        preds.append(y_pred.item())
        targets.append(y_val[i])
    
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,2))
    targets = scaler.inverse_transform(np.array(targets).reshape(-1,2))

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets,preds)
    r2 = r2_score(targets, preds)
    diff_percentage = []
    print("Size of preds and targets: ", preds.size, targets.size)
    for i in range(preds.size//2):
        diff = 100*(preds[i,0] - targets[i,0])/(targets[i,0]) #100*abs(((abs(preds[i]) - abs(targets[i])) / (abs(preds[i])) - abs(targets[i])) / 2) # Absolute Percentage Difference
        diff_percentage.append(diff)


    #Perhaps move results to separate visualization function
    #Add percentage error function with minus
    #print(f"Validation MSE: {mse:.6f}")
    #print(f"Validation MAE: {mae:.6f}")
    #print(f"Validation R2: {r2:.4f}")
    print("Size of Training Data: ", train_data.size)
    print("Number of Epochs: ", epochs)
    print(f"Mean Percentage Difference: {np.mean(diff_percentage):.2f}%")
    print("")

    print("Predictions : Actual : Percentage Difference")
    print(np.c_[preds[:,0],targets[:,0], diff_percentage])

    graphing(diff_percentage, preds[:,0], targets[:,0])

    # --- Predict next 24 steps using the last available validation sequence ---
    last_sequence = X_val[0]  # shape: (seq_len, 1, 1)
    future_preds = predict_future(lstm, W_out, b_out, last_sequence, scaler, steps=24)
    f_diff_percentage = []
    for i in range(future_preds.size//2):
        index = i
        f_diff = 100*(future_preds[i,0] - targets[index,0])/(targets[index,0]) #100*abs(((abs(preds[i]) - abs(targets[i])) / (abs(preds[i])) - abs(targets[i])) / 2) # Absolute Percentage Difference
        f_diff_percentage.append(f_diff)

    print("\nFuture Predictions for the Next 24 Steps:")
    print(future_preds[:,0].flatten())
    print("Percentage difference:", np.c_[f_diff_percentage])
    print (f"Mean Percentage Difference: {np.mean(f_diff_percentage):.2f}%")
    graphing(f_diff_percentage, future_preds[:,0], targets[:12,0])

    if (write_to_file == True):
        sys.stdout.close()










