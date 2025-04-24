import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import numpy as np

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


def generate_data(seq_len = 50, total_points = 1000): #sine wave data
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


hidden_dim = 10
input_dim = 1
seq_len = 25
learning_rate = 0.01
epochs = 5

lstm = LSTMCell(input_dim, hidden_dim)
W_out = np.random.randn(1, hidden_dim) * 0.1 #Output layer weights
b_out = np.zeros((1,1))

X,y = generate_data(seq_len)

for epoch in range(epochs):
    loss_epoch = 0
    for i in range(len(X)):
        x_seq = X[i]
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










