import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

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
        self.Wf = np.random.randn(hidden_dim,input_dim + hidden_dim) * 0.1
        self.Wi = np.random.randn(hidden_dim,input_dim + hidden_dim) * 0.1
        self.Wo = np.random.randn(hidden_dim,input_dim + hidden_dim) * 0.1
        self.Wc = np.random.randn(hidden_dim,input_dim + hidden_dim) * 0.1

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
        candidate_cell_gate = sigmoid(np.dot(self.Wc, concat) + self.bc)
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


def create_sequences(data, seq_length = 25):
    X,y = [],[]
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].reshape(-1,1,1))
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def run_model(train_data, val_data, scaler, epochs = 5, learning_rate = 0.01):
    hidden_dim = 10
    input_dim = 1
    seq_len = 25
    
    

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
            error = y_pred - y[i]
            loss = error ** 2
            loss_epoch += loss.item()

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
    
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    targets = scaler.inverse_transform(np.array(targets).reshape(-1,1))

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets,preds)
    r2 = r2_score(targets, preds)
    diff_percentage = []
    print("Size of preds and targets: ", preds.size, targets.size)
    for i in range(preds.size):
        diff = 100*np.abs(preds[i] - targets[i])/np.abs(targets[i]) #100*abs(((abs(preds[i]) - abs(targets[i])) / (abs(preds[i])) - abs(targets[i])) / 2) # Absolute Percentage Difference
        diff_percentage.append(diff)
    #Perhaps move results to separate visualization function
    #Add percentage error function
    print("")
    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation MAE: {mae:.6f}")
    print(f"Validation R2: {r2:.4f}")
    print(f"Mean Percentage Difference: {np.mean(diff_percentage):.2f}%")
    print("")

    print("Predictions : Actual : Percentage Difference")
    print(np.c_[preds,targets, diff_percentage])










