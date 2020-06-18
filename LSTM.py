import numpy as np
from matplotlib import pyplot
import pandas as pd
from keras.models import Sequential                
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
import time
from numpy import newaxis
from sklearn import datasets, metrics, preprocessing
#variabili globali

unnormalized_price = []
means = []
std = []
def readCSV():
    #decommentare questo!
    #stocks = pd.read_csv('WIKIFB.csv', header=None)
    stocks = pd.read_csv('YAHOO.csv', header=None)
    df = pd.DataFrame(stocks)
    #elimino il campo data
    #decommentare questo!
    #df.drop(df.columns[0], axis = 1, inplace = True)
    #O/H/L/C/V/AC WIKI-FB.csv:
    #df.drop(df.columns[[0,6,7,8,9,10,12]], axis = 1, inplace = True)
    #O/H/L/C/V/AC YAHOO.csv:
    df.drop(df.columns[0], axis = 1, inplace = True)
    #print(df)
    return df
#readCSV()   

#formatto i dati raggruppandoli per batch col metodo a finestra

def load_data(stocks,batch_len,normalization):
    global unnormalized_price
    global means
    global std
    batch_row_num = batch_len + 1 
    col_number = len(stocks.columns)
    #converto da dataframe a matrice (altrimenti ci sono problemi)
    stocks = stocks.as_matrix();
    result = []
    #appendo i primi batch_num elementi ogni volta shiftati di 1
    for i in range(len(stocks) - batch_row_num +1):
        result.append(stocks[i+1: i + batch_row_num])
   
    result = np.array(result)
    if normalization:
        result = batch_normalization(result)
        #result = z_score_standardization(result)
    #result 913 x 50 x col_number
    #print(result)
    #print(len(result[0, :, :]))
    #print(result.shape[0])
    train_len = round(0.9 * len(result))
    #print(len(result))
    train = result[:int(train_len),:]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    #sto provando a predirre tramite i primi 49 elementi il 50 esimo
    #considero solo il campo all'indice 10, corrispondente ad adj.close
    for i in range(int(train_len)):
        #decommentare questi:
            # x_train.append(train[i,:-1,10])
            # y_train.append(train[i,-1:,10])
            x_train.append(train[i,:-1,:])
            #decommentare questo per adj.close di WIKI-FB.csv
            #y_train.append(train[i,-1:,5])
            y_train.append(train[i,-1:,4])
        
    #ho preso l'ultimo elemento per ogni campione come target 
    #print(train)
    unnormalized_price = unnormalized_price[int(train_len):]
    means = means[int(train_len):]
    std = std[int(train_len):]
    #print(len(unnormalized_price))
    #decommentare questi:
    # x_test  = result[int(train_len):,:-1,10]
    # y_test = result[int(train_len):,-1,10]
    x_test  = result[int(train_len):,:-1,:]
    #decommentare questo per adj.close di WIKI-FB.csv
    #y_test = result[int(train_len):,-1,5]
    y_test = result[int(train_len):,-1,4]
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #print(y_test.shape[0])
    #i layer delle LSTM in keras accettano come input (N, W, F) 
    #3D tensor with shape (batch_size, timesteps, input_dim), (Optional) 2D tensors with shape (batch_size, output_dim).
    #faccio un reshape di x_train e x_test
    #decommentare questi:
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], col_number))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], col_number))  
    return [x_train, y_train, x_test, y_test]

    #normalizzo i batch con la formula p(i)/p(0) - 1 per ogni finestra
#decommentare questo
# def batch_normalization(batch):
#     for i in range(batch.shape[0]):
#         adj_close_start = batch[i,0,10]
#         unnormalized_price.append(adj_close_start)
#         for j in range(batch.shape[1]):
#             batch[i,j,10] = (float(batch[i,j,10]) / float(adj_close_start)) - 1
#     return batch
    
    #denormalizzo i batch alla fine del processo di predizione 

def batch_normalization(batch):
    adj_close_start = []
    for i in range(batch.shape[0]):
        adj_close_start[:] = batch[i,0,:]
        #decommentare questo per adj.close di WIKI-FB.csv:
        #unnormalized_price.append(adj_close_start[5])
        unnormalized_price.append(adj_close_start[4])
        for j in range(batch.shape[1]):
            for k in range(batch.shape[2]):
                batch[i,j,k] = (float(batch[i,j,k]) / float(adj_close_start[k])) - 1
    return batch
#decommentare questo:

# def batch_denormalization(predicted_val):
#     for i in range(predicted_val.shape[0]):
#         #unnormalized_price[0] = float(unnormalized_price[0])
#         predicted_val[i] = float(unnormalized_price[i])*(predicted_val[i]+1)
#     return predicted_val
 
def batch_denormalization(predicted_val):
    for i in range(predicted_val.shape[0]):
        #unnormalized_price[0] = float(unnormalized_price[0])
        predicted_val[i] = float(unnormalized_price[i])*(predicted_val[i]+1)
    return predicted_val
    #for i in range(predicted_val.shape[0]):
    
        #predicted_val[i] = float(unnormalized_price[i]) * (predicted_val[i] + 1)

def z_score_standardization(batch):

    for i in range(batch.shape[0]):
        #batch[i,:,10] = preprocessing.StandardScaler().fit_transform(batch[i,:,10])
        scale = preprocessing.StandardScaler().fit(batch[i,:,10])
        means.append(scale.mean_)
        std.append(scale.std_)
        batch[i,:,10] = scale.transform(batch[i,:,10])        #for j in range(batch.shape[1]):
            #batch[i,j,10] = (batch[i,j,10]) - np.var(batch[i,:,10])) / np.std(batch[i,:,10])
        #print(batch)
    return batch

def z_score_denormalization(batch):
    for i in range(batch.shape[0]):
        batch[i] = batch[i]*std[i] + means[i]
    return batch

#d = readCSV()
#load_data(d,3)


def build_model(layers):
    model = Sequential()
    #50, 13
    model.add(LSTM(
        #layers[1],
        input_dim = layers[0],
        #input_shape=(layers[1]-1,layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[3],
        return_sequences=False))
    model.add(Dropout(0.2))

#The last layer we use is a Dense layer ( = feedforward). Since we are doing a regression, its activation is linear.
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

#Adam, algoritmo di discesa stocastica del gradiente (SGD), semplificazione dell'agoritmo di discesa del gradiente
def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        #Accuracy" is defined when the model classifies data correctly according to class, 
        #but "accuracy" is effectively not defined for a regression problem, due to its continuous property.
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model


#prediction 
def getUnnormalizedPrice():
    print unnormalized_price

#metrica
def calculate_Accuracy(p,t):
    acc = []
    for i in range(p.shape[0]):
        acc.append((p[i] - t[i])/t[i])
    acc = np.array(acc)
    acc = np.mean(abs(acc), dtype = np.float64)
    return acc

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #predice la sequenza di 50 step prima di shiftare di 50 step avanti
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
