import LSTM1
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics, preprocessing
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [0 for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
	global_start_time = time.time()
	#ere
	epochs  = 250
	seq_len = 22

	print('> Loading data... ')
	d = LSTM1.readCSV()
	X_train, y_train, X_test, y_test = LSTM1.load_data(d, seq_len, True)
	#print(X_train)
	#print(y_train.shape)
	#print(X_test)
	#print(y_test.shape)
	#print(y_train.shape)
	print('> Data Loaded. Compiling...')
	#LSTM1.batch_denormalization()
	#model = LSTM1.build_model([13, 50, 100, 1])
	#model = LSTM1.build_model([13, 50, 100, 13])
	model = LSTM1.build_model2([6, 21, 1])
	model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    epochs=epochs,
	    validation_split=0.05)

	predictions = model.predict(X_test)

	#print(predictions)
	#predictions = LSTM1.predict_sequences_multiple(model, X_test, seq_len-1, 50)
	#print(predictions)
	predictions = np.array(predictions)
	#LSTM1.getUnnormalizedPrice()
	#valori di predizione denormalizzati
	predictions = LSTM1.batch_denormalization(predictions)
	#predictions = LSTM1.z_score_denormalization(predictions)
	y_test_denormalized = []
	y_test_denormalized = LSTM1.batch_denormalization(y_test)
	#y_test_denormalized = LSTM1.z_score_denormalization(y_test)
	acc = LSTM1.calculate_Accuracy(predictions,y_test_denormalized)
	print('Evaluation: ',acc)
	score = metrics.mean_squared_error(predictions, y_test_denormalized)
	#print(predictions)
	#print(y_test_denormalized)
	print ("MSE: %f" % score)
	r2 = metrics.r2_score(y_test_denormalized,predictions)
	print('R2 SCORE: ',r2)
	plt.plot(predictions,color='red', label='Valori predetti normalizzati')
	plt.plot(y_test_denormalized,color='blue', label='Valori y_test normalizzati')
	plt.legend(loc='upper left')
	plt.show()
	#print(y_test_denormalized)
	#print(predictions)   
	print('Training duration (s) : ', time.time() - global_start_time)
	#plot_results_multiple(predictions, y_test_denormalized, 50)