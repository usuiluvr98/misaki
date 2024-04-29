import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = {
    'X1':[1,2,3,4,5],
    'X2':[1,4,12,16,20],
    'Y':[1,2,4,6,8]
}
df = pd.DataFrame(data)

X = df[['X1','X2']]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

print("RMSE = ",mean_squared_error(Y_test,Y_pred,squared=False))
print("Coefficients = ", model.coef_)
print("Intercept = ", model.intercept_)

diff = pd.DataFrame({'Actual value': Y_test, 'Predicted value': Y_pred})
print('\n')
print(diff.head())