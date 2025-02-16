import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Track error during training
error_track = list()
error_track2 = list()


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

        # Track log-loss
        loss = log_loss(y, predictions)
        error_track.append(loss)

        # Check for convergence
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    return coef


def SGD(X, y, coef, learning_rate):
    for i in range(len(X)):
        predictions = sigmoid(X[i], coef)
        error = y.iloc[i] - predictions
        gradient = np.dot(X[i], error)  # Stochastic gradient descent
        coef += learning_rate * gradient  # Update coefficient

        # Track log-loss over the entire dataset
        predictions_all = sigmoid(X, coef)
        loss = log_loss(y, predictions_all)
        error_track2.append(loss)
    return coef


def model_fit(X, y, epochs):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    coef_init = np.zeros(X_bias.shape[1])  # Initialize coefficients

    # Train using gradient ascent
    final_coef = gradient_ascent(X_bias, y, coef_init.copy(), learning_rate=0.1, epochs=epochs)

    # Reinitialize coefficients for SGD
    final_coef2 = SGD(X_bias, y, coef_init.copy(), learning_rate=0.1)
    return final_coef, final_coef2


def predict(X, coef):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    return np.round(sigmoid(X_bias, coef))


# Load and preprocess data
data = pd.read_csv('data/framingham.csv')
data = data.dropna()
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the model
epochs = 1000
coef, coef2 = model_fit(X_train, y_train, epochs=epochs)
print("\nGradient Ascent Coefficients:\n", coef)
print("\nStochastic Gradient Descent Coefficients:\n", coef2)

# Predict on test set
y_pred = predict(X_test, coef)
y_pred_sgd = predict(X_test, coef2)

# Compare with scikit-learn's LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
print("\nScikit-learn Coefficients:\n", lr.coef_, lr.intercept_)
y_pred2 = lr.predict(X_test)

# Evaluate precision
print("\nGradient Ascent Precision:", precision_score(y_test, y_pred))
print("Stochastic Gradient Descent Precision:", precision_score(y_test, y_pred_sgd))
print("Scikit-learn Precision:", precision_score(y_test, y_pred2))

# Plot training error
plt.figure(figsize=(12, 6))
plt.plot(range(len(error_track)), error_track, label='Gradient Ascent')
plt.plot(range(len(error_track2)), error_track2, label='SGD')
plt.xlabel('Iterations')
plt.ylabel('Log-Loss')
plt.title('Training Error Over Iterations')
plt.legend()
plt.show()

