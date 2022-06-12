import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read csv into a Dataframe
house_data = pd.read_csv('../Datasets/Datasets/house_prices.csv')
size = house_data['sqft_living']
price = house_data['price']

x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# use linear regression
model = LinearRegression()
model.fit(x, y)

# MSE ane
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x, y))

# b1
print(model.coef_[0])
# b0
print(model.intercept_[0])

plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

# Predicting the prices
print("Prediction by the model: ", model.predict([[2000]]))