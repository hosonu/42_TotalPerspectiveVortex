import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic regression classifier implemented from scratch (bonus task).
    Optimizes weights and bias using gradient descent.
    """
    def __init__(self, learning_rate=0.05, n_iterations=1000, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Clip inputs to avoid overflow in exp
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Train with gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.lambda_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        return self

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Threshold at 0.5 for class labels (0 or 1)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)