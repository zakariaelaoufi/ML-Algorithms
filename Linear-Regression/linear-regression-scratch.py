import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Batch gradient descent
def gradient_descent(X, y, learning_rate, epochs, intercept, coefficient):
    m = len(X)
    for i in range(epochs):
        y_pred = intercept + np.dot(X, coefficient)
        error = y - y_pred
        intercept += (learning_rate / m) * np.sum(error)
        coefficient += (learning_rate / m) * np.dot(X.T, error)
        # print(np.sum(error))
    return intercept, coefficient


def linear_regression(X, y, learning_rate, epochs):
    print(X.shape, y.shape)
    n_features = X.shape[1]
    print(n_features)
    intercept_0 = np.random.normal(0, 1)
    coefficient_0 = np.random.normal(0, 1, n_features)
    intercept, coefficient = gradient_descent(X, y, learning_rate, epochs, intercept_0, coefficient_0)
    return intercept, coefficient


# Normal equation
def normal_equation(X, y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    return theta[0], theta[1:]


# import data
data = pd.read_csv('../data/Salary_dataset.csv')

# Example data
X = data['YearsExperience'].to_numpy().reshape(-1, 1)  # Reshape X to 2D array
y = data['Salary'].to_numpy()

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Run linear regression
intercept, coefficient = linear_regression(X, y, learning_rate, epochs)

print("Intercept:", intercept)
print("Coefficients:", coefficient)

intercept1, coefficient1 = normal_equation(X, y)
print('intercept using normal equation:', intercept1)
print('coefficient using normal equation:', coefficient1)


ln = LinearRegression()
ln.fit(X, y)
print("Intercept:", ln.intercept_)
print("Coefficients:", ln.coef_)
