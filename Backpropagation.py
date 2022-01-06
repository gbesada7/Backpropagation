import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score # for cross-validation
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns

#Neural network backpropagation
#sigmoid function 1/(1+(e^-net)) where net = sum of (w(i))*(x(i)) --> sum of the weights 

# 3 layers from left to right
#left is input layer, next is hidden layer, then is output layer
#input is connected to all hidden layers

# once get output going to adjust the weights starting from right to left (from hidden to input layer)
# based on the rate of error...so thats why backpropagation is backwards it goes from output to hidden to input layer

#adjust weights in order to obtain smallest error possible using gradient descent algorithm
#gradient descent: 


# read in the file

data = pd.read_csv("data.csv")
#print(data.head())
y = data["Rejects"].values

X = data[["Temp", "Pres", "Flow", "Process"]].values

results = pd.DataFrame(columns = ["mse", "accuracy"])


#check if columns and rows match, otherwise train_test_split will not work

#print("shape of X: " + str(X.shape[0]))
#print("shape of y: " + str(y.shape[0]))
#print(y.astype(str))


#split dataset into 80-20 (reduces bias)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

#print(X_test)
#print(X_train)
#print(y_test)
#print(y_train)
#print("shape of X_train" + str(X_train.shape))
#print("shape of X_test" + str(X_test.shape))

#Scale values...MinMaxScaler will scale values between 0 and 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#check if values are 0 and 1
print(X_train.min())
print(X_train.max())


from tensorflow.keras.models import Sequential #layers
from tensorflow.keras.layers import Dense #layers

#help(Dense)

#normal feedforward network
# neurons connected to first layer denseley connected (i.e every neuron connected to other)
# rectified linear unit
model = Sequential([Dense(21, activation = 'relu'), Dense(21, activation = 'relu'), Dense(1)]) 

#gradient descent implementation
model.compile(optimizer = 'rmsprop', loss = 'mse')

model.fit(x = X_train, y = y_train, epochs = 250)

#print(model.history.history)

# will consist of the change of mean squared error values (should decrease which each step and converge)
loss_df = pd.DataFrame(model.history.history)

# plot the mean squared error values
#loss_df.plot()


mean_squared = model.evaluate(X_test, y_test, verbose = 0)

mean_squared2 = model.evaluate(X_train, y_train, verbose = 0)

#predict the model values
test_predictions = model.predict(X_test)

test_predictions = pd.Series(test_predictions.reshape(60, ))

pred_df = pd.DataFrame(y_test, columns = ['Test True Y'])

pred_df = pd.concat([pred_df, test_predictions], axis = 1)

pred_df.columns = ['Test True Y', 'Model Predictions']

# will plot the actual values against the predicted ones (ideally should be a straight line if we are accurate)
sns.scatterplot(x = 'Test True Y', y = 'Model Predictions', data = pred_df)


#failed attempt
"""
from sklearn.metrics import mean_squared_error

Mean_squared3 = mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])*0.5
"""
"""

neural_network = MLPClassifier(hidden_layer_sizes = (10, 10, 10, 10), max_iter = 5000) 

perceptron = neural_network.fit(X_train, y_train.astype(str))

predictions = neural_network.predict(X_test)

a = y_test.values


count = 0
for i in range(len(pred)):
    if pred[i] == a[i]:
        print(pred[i])
        print(a[i])
        count = count + 1

print(count)
print(len(pred))

print(count/len(pred))

print(confusion_matrix(y_test, int(predictions)))
score = cross_val_score(neural_network, X_train.astype(str), y_train.astype(str)).mean() # mean cross-validation accuracy
print(f'Mean cross-validation accuracy MLP = {score:0.4f}')

"""
