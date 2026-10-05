import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv("ex1data2.txt", header=None)
print(data.head())

X = data.iloc[:, 0:2] # take evey row of column 0 to 1 (excluding 2)
y = data.iloc[:, 2]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42 )

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE is: {mse:.4f}")
print(f"RMSE (better for dollars) is: {rmse:.4f}")
#off by apprx. $87,241

