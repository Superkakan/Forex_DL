#import tensorflow as tf
#from tensorflow import keras
#from keras import layers, Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

from presentation.graph import graphing 

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
        #weight vectors
        self.Wf = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        self.Wi = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        self.Wo = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        self.Wc = np.random.uniform(-limit, limit, (hidden_dim, input_dim + hidden_dim))
        #bias vectors
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

def create_sequences(data, seq_length = 32, pred_steps = 1):
    X,y = [],[]
    for i in range(len(data) - seq_length - pred_steps):
        sequence = data[i:i + seq_length]
        label = data[i + seq_length + pred_steps - 1,0]
        X.append(sequence.reshape(seq_length, data.shape[1], 1)) #1 
        y.append(label)
    #(sequence_length, num_features, 1)
    
    y = np.array(y)
    print("Labels shape: ",y.shape)
    return np.array(X), np.array(y)


def predict_future(model, W_out, b_out, initial_seq, scaler, steps=24): # More of a helper function
    lstm = model
    hidden_dim = lstm.hidden_dim
    predictions = []


    current_seq = initial_seq.copy()  # (seq_len, 1, num_features)
    num_features = current_seq.shape[1]

    for _ in range(steps):
        h = np.zeros((hidden_dim, 1)) # Shape of the array ofc
        c = np.zeros((hidden_dim, 1))

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
        new_input = new_step.reshape(1, num_features, 1) # Batch, timesteps available (historical datapoints), features 
        # Batch/sequence_length, nr_features, just a one (timesteps)
        # Append new input and remove oldest    
        current_seq = np.concatenate([current_seq[1:], new_input], axis=0)
    
    return current_seq #predictions_inverse



def run_model(train_data, val_data, scaler, pred_step, epochs = 5, learning_rate = 0.01,write_to_file = False):
    hidden_dim = 128
    input_dim = train_data.shape[1]
    seq_len = 32

    
    if (write_to_file == True):
        sys.stdout = open("results.txt", "a")
        print("")

    lstm = LSTMCell(input_dim, hidden_dim)
    W_out = np.random.randn(1, hidden_dim)# * 0.1 #Output layer weights
    b_out = np.zeros((1,1))
    X,y = create_sequences(train_data, seq_len, pred_step) # Training data
    X_val, y_val = create_sequences(val_data, seq_len,pred_step)
    # Check first 3 sequences
    #for i in range(3):
    #    seq = X[i][:,0,0]  # extract the first feature from each step
    #    target = y[i]
    #    print(f"Sequence {i}: {seq}")
    #    print(f"Target {i}: {target}")
    #    print(f"Next value after last in sequence? {'Yes' if abs(seq[-1] - target) > 0 else 'No'}\n")

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
            target = y[i] #np.array(y[i])
            # L2 Loss
            error = y_pred - target 
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

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets,preds)
    r2 = r2_score(targets, preds)
    diff_percentage = []
    print("Size of preds and targets: ", len(preds), len(targets))#targets.size
    for i in range(len(preds)): #preds.size//2
        diff = 100*(preds[i] - targets[i])/(targets[i]) 
        diff_percentage.append(diff)
    
    combined_preds = [ [i, 0] for i in preds]
    preds_inversed = scaler.inverse_transform(combined_preds)
    preds_real = [i[0] for i in preds_inversed]

    combined_targets_full = [ [i, 0] for i in targets]
    targets_full_inversed = scaler.inverse_transform(combined_targets_full)
    combined_target_full_real = [i[0] for i in targets_full_inversed]

    graphing(diff_percentage, preds_real, combined_target_full_real)


    #Predict next steps using the last available validation sequence
    
    last_sequence = X_val[0]
    future_preds = predict_future(lstm, W_out, b_out, last_sequence, scaler, pred_step) # Denormalized in function


    print(f"Validation MSE: {mse:.6f}")
    print(f"Validation MAE: {mae:.6f}")
    print(f"Validation R2: {r2:.4f}")
    print("Size of Training Data: ", train_data.size)
    print("Number of Epochs: ", epochs)
    print(f"Mean Percentage Difference: {np.mean(diff_percentage):.2f}%")
    print("")


    combined_preds_fut = [ [i[0][0], 0] for i in future_preds] # the i[0][0] shouldnt be needed, somewhere in the future_predictions function something changes
    preds_inversed_fut = scaler.inverse_transform(combined_preds_fut)
    preds_real_fut = [i[0] for i in preds_inversed_fut]
    future_preds = preds_real_fut

    
    combined_targets = [[t, 0] for t in targets]
    targets_inverse = scaler.inverse_transform(combined_targets)
    targets_real = [row[0] for row in targets_inverse]





    f_diff_percentage = []
    for i in range(len(future_preds)):
        f_diff = 100*(future_preds[i] - targets_real[i])/(targets_real[i])
        f_diff_percentage.append(f_diff)

    #print(f"\nFuture Predictions for the Next {pred_step} Steps:")
    #print(future_preds[:-pred_step])
    #print("Target Values:")
    #print(targets_real[:-pred_step])
    #print("Percentage difference:", np.c_[f_diff_percentage])
    print (f"Mean Percentage Difference: {np.mean(f_diff_percentage):.2f}%")
    graphing(f_diff_percentage, future_preds[:-pred_step], targets_real)

    if (write_to_file == True):
        sys.stdout.close()










