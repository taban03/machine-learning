import numpy as np
from matplotlib import pyplot
import pandas as pd
from keras.models import Sequential                
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
#a = [1,2, 3,4]
#print(a[:,-1])

def readCSV():
    stocks = pd.read_csv('WIKIFB.csv', header=None)
    df = pd.DataFrame(stocks)
    return df
#readCSV()   

#formatto i dati raggruppandoli per batch col metodo a finestra

def load_data(stocks, batch_len,normalization):
    batch_row_num = batch_len + 1 
    col_number = len(stocks.columns)
    #converto da dataframe a matrice (altrimenti ci sono problemi)
    stocks = stocks.as_matrix();
    #print(len(stocks))
    result = []
    #appendo i primi batch_num elementi ogni volta shiftati di 1
    for i in range(len(stocks) - batch_row_num +1):
        result.append(stocks[i+1: i + batch_row_num])

    
        
    result = np.array(result)
    if batch_normalization:
        result = batch_normalization(result)
    #result 959 x 3 x 13
    print(result)
    print(len(result[0, :, :]))
    print(result.shape[0])
    train_len = round(0.9 * len(result))
    #print(len(result))
    train = result[:int(train_len),:]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    #sto provando a predirre tramite i primi 49 elementi il 50 esimo
    for i in range(int(train_len)):
            x_train.append(train[i,:-1,:])
            y_train.append(train[i,-1:,:])

    #x_train = result[:, :-1]
    #y_train = result[:, 
    #ho preso l'ultimo elemento per ogni campione come target 
    #print(train)

    x_test  = result[int(train_len):,:-1]
    y_test = result[int(train_len):,-1]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #print(x_train)
    #print("djshsh")
    #print(x_test)
    #i layer delle LSTM in keras accettano come input (N, W, F) 
    #3D tensor with shape (batch_size, timesteps, input_dim), (Optional) 2D tensors with shape (batch_size, output_dim).
    #faccio un reshape di x_train e x_test
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 13))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 13))  
    return [x_train, y_train, x_test, y_test]

    #normalizzo i batch con la formula p(i)/p(0) - 1 per ogni finestra

def batch_normalization(batch):
    for i in range(batch.shape[0]):
        adj_close_start = batch[i,0,11]
        for j in range(batch.shape[1]):
            batch[i,j,11] = (float(batch[i,j,11]) / float(adj_close_start)) - 1

    return batch
    

#d = readCSV()
#load_data(d,3)

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

   
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model
