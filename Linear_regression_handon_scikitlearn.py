### IMPORTANT NOTICE ###
# NO GENERATIVE AI WAS USED IN THE CODE 
# Sources used - StackOverflow, datacamp, Handson-1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("food_truck.txt", sep=",", header=None) #Header is none because there are no titles in the txt file
X = data.iloc[:, 0]
print(X.head())
print(len(X))
y = data.iloc[:, 1]
print(len(y))
X = X.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray
y = y.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray


#Making a train test split 
X_train, X_test, y_train, y_test = train_test_split (X,y, random_state=42,test_size=0.3) #random_state=42, just because it is the ultiamte answer, test_size = 0.3 so the test set will be 30%

#Importing Linear Regression and fitting the model with training data
model = LinearRegression()
model.fit(X_train,y_train)

#Making Predicitons
y_pred = model.predict(X_test)

#Evaluating the model
mse = mean_squared_error(y_test,y_pred)
print(f"MSE is: {mse:.4f}")

#Plotting the model
plt.scatter(X[:,0], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,0], model.predict(X), color = 'red') # we use model_predict X so both x and y have the same size
plt.show()



#results are not the same because here a train test split is done, which means that the model trains on 70% on the data and tests on 30% of it