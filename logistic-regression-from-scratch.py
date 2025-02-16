import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


error_track = list()

def sigmoid(X, coef):
    z = np.dot(X, coef)
    return np.clip(1 / (1 + np.exp(-z)), 1e-15, 1 - 1e-15)


def gradient_ascent(X, y, coef, learning_rate, epochs, tol=1e-4):
    prev_loss = float('inf')
    for _ in range(epochs):
        predictions = sigmoid(X, coef)
        error = y - predictions
        gradient = np.dot(X.T, error) / X.shape[0]  # Average gradient
        coef += learning_rate * gradient  # Gradient ascent

        # Optional: Check convergence using log-loss
        loss = log_loss(y, predictions)
        error_track.append(loss)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return coef


def model_fit(X, y, epochs):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    coef = np.zeros(X_bias.shape[1])  # Initialize to zeros
    # coef = np.random.normal(0, 1, X_bias.shape[1])  # Initialize coefficients
    final_coef = gradient_ascent(X_bias, y, coef, learning_rate=0.1, epochs=epochs)
    return final_coef


def predict(X, coef):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    return np.round(sigmoid(X_bias, coef))

################################################

# Load and preprocess data
data = pd.read_csv('data/framingham.csv')
data = data.dropna()

X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# Standardize features (excluding bias term added later)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

################################################
# Fit the model
epochs = 1000
coef = model_fit(X_train, y_train, epochs=epochs)
print("\n",coef)
y_pred = predict(X_test, coef)

################################################
lr = LogisticRegression()
lr.fit(X_scaled, y)
print("\n",lr.coef_, lr.intercept_)
y_pred2 = lr.predict(X_test)

###############################################
from sklearn.metrics import precision_score

print(precision_score(y_test, y_pred))
print("-------------------------")
print(precision_score(y_test, y_pred2))


###############################################
plt.plot(error_track, range(len(error_track)))
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
