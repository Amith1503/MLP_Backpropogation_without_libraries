import numpy as np
import pandas as pd 
import pickle as pkl

def test_mlp(data_file):

	test_data=pd.read_csv(data_file);


	def sigmoid(data):
		return (1/(1+np.exp(-data)))

	def softmax(data):
		exponential=np.exp(data)
		return exponential / exponential.sum(axis=1, keepdims=True)

	def round_y_pred(y_pred):
		zeros_ypred=np.zeros((len(y_pred),4))
		ypred_index=np.argmax(y_pred,axis=1)
		for i in range (len(y_pred)):
			zeros_ypred[i][ypred_index[i]]=1

		return zeros_ypred


	def predict(X_test,W1,b1,W2,b2):
		Z1 = np.dot(X_test, W1) + b1
		A1 = sigmoid(Z1)
		Z2 = np.dot(A1, W2) + b2
		pred = softmax(Z2)
		y_predict= round_y_pred(pred)
		return y_predict

	pickle_dict= pkl.load(open("model_weights.pkl", "rb"))
	y_pred=predict(test_data,pickle_dict["W1"],pickle_dict["b1"],pickle_dict["W2"],pickle_dict["b2"])

	return y_pred
    
    

    

'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''
