import numpy as np
from BaseModel import ClassificationModel


class Perceptron(ClassificationModel):
    def __init__(self, alpha: float, epochs: int = 50, seed: int = None, verbose=True, debug=True):
        super().__init__()
        self.alpha = alpha  # Learning rate, determines the step size during the gradient descent
        self.epochs = epochs  # Number of iterations
        self.seed = seed  # Seed for random number generator
        self.verbose = verbose  # If True, print logs
        self.debug = debug  # If True, print additional detailed debugging
        if self.seed:
            np.random.seed(self.seed)
        self.W = None  # Weights of the perceptron
        self.b = None  # Bias of the perceptron

    def initialize_weights(self, n_features):
        # Initialize weights randomly, +1 for bias
        self.W = np.random.uniform(-0.05, 0.05, n_features + 1)

    def activation(self, z):
        return 1 if z > 0 else -1  # Activation function, here it's a step function

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)  # Adding bias unit to the input

        for epoch in range(self.epochs):
            for idx in range(n_samples):
                xi = X[idx]
                yi = y[idx]
                prediction = self.activation(
                    np.dot(xi, self.W))  # Compute the prediction
                error = yi - prediction  # Compute the error
                self.W += self.alpha * error * xi  # Update the weights

                if self.debug and (idx % 100 == 0):
                    print(
                        f"Epoch {epoch+1}, Sample {idx}: Prediction={prediction}, Actual={yi}")

    def predict(self, X: np.ndarray):
        # Ensure bias unit is added to the input
        X = np.insert(X, 0, 1, axis=1)
        predictions = np.dot(X, self.W)  # Compute the predictions
        # Apply the activation function to each prediction
        return np.array([self.activation(x) for x in predictions])
