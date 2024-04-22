import numpy as np
from BaseModel import ClassificationModel


class Perceptron(ClassificationModel):
    def __init__(self, alpha: float, epochs: int = 50, seed: int = None, verbose=True, debug=True):
        super().__init__()
        self.alpha = alpha  # Learning rate
        self.epochs = epochs  # Number of epochs
        self.seed = seed  # Seed for random number generator
        self.verbose = verbose  # Print logxs
        self.debug = debug  # Additional detailed debugging
        if self.seed:
            np.random.seed(self.seed)
        self.W = None  # Weights
        self.b = None  # Bias

    def initialize_weights(self, n_features):
        self.W = np.random.uniform(-0.05, 0.05, n_features + 1)  # +1 for bias

    def activation(self, z):
        return 1 if z > 0 else 1

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.insert(X, 0, 1, axis=1)  # Adding bias unit

        for epoch in range(self.epochs):
            for idx in range(n_samples):
                xi = X[idx]
                yi = y[idx]
                prediction = self.activation(np.dot(xi, self.W))
                error = yi - prediction
                self.W += self.alpha * error * xi

                if self.debug and (idx % 100 == 0):
                    print(
                        f"Epoch {epoch+1}, Sample {idx}: Prediction={prediction}, Actual={yi}")

    def predict(self, X: np.ndarray):
        X = np.insert(X, 0, 1, axis=1)  # Ensure bias unit is added
        predictions = np.dot(X, self.W)
        return np.array([self.activation(x) for x in predictions])
