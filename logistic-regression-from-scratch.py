import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def sigmoid(X, coef):
    return 1 / (1 + np.exp(-np.dot(X, coef)))


def gradient_ascent(X, y, coef, learning_rate, epochs):
    for _ in range(epochs):
        prediction = sigmoid(X, coef)
        error = y - prediction
        coef += learning_rate * np.dot(X.T, error)
    return coef


def model_fit(X, y):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    coef = np.random.normal(0, 1, X_bias.shape[1])  # Initialize coefficients
    final_coef = gradient_ascent(X_bias, y, coef, learning_rate=0.05, epochs=500)
    return final_coef

def predict(X, coef):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    return np.round(sigmoid(X_bias, coef))


# Load and clean data
data = pd.read_csv('data/framingham.csv')
data = data.dropna()  # Remove rows with missing values

X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize the features

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the model
coef = model_fit(X_train, y_train)
print(coef)

y_pred = predict(X_test, coef)

#################################################"
lr = LogisticRegression()

lr.fit(X_scaled, y)

print(lr.coef_, lr.intercept_)

y_pred2 = lr.predict(X_test)

###############################################
from sklearn.metrics import precision_score

print(precision_score(y_test, y_pred))
print("tets")
print(precision_score(y_test, y_pred2))

