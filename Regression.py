import matplotlib.pyplot as plt
import numpy as np
from BaseModel import ClassificationModel
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import seaborn as sns

# Define the Logistic Regression class, which inherits from the ClassificationModel class


class LogisticRegression(ClassificationModel):
    # Initialize the class with learning rate (alpha), number of epochs, random seed, and batch size
    def __init__(self, alpha: float, epochs: int = 1, seed: int = None, batch_size: int = None):
        ClassificationModel.__init__(self)
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.w = None  # Initialize weights to None

    # Method to plot and save the confusion matrix
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> None:
        y_pred = self.predict(X)
        conf_mat = confusion_matrix(y, y_pred)

        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(
            '/Users/kevinhudson/Documents/Classification Mini Project/confusion_matrix_regression.png')
        plt.show()

    # Softmax activation function for multi-class classification
    def softmax(self, z: np.ndarray) -> np.ndarray:
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)

    # Method to train the model
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        y_one_hot = np.eye(n_classes)[y].reshape(-1, n_classes)

        if self.batch_size is None:
            self.batch_size = n_samples

        # Initialize weights randomly
        self.w = np.random.randn(n_features, n_classes)
        previous_loss = np.inf
        loss_threshold = 0.0075  # 5% threshold for adjusting alpha

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y_one_hot[i:i + self.batch_size]

                z = X_batch.dot(self.w)
                probs = self.softmax(z)

                # Compute gradient and update weights
                grad = X_batch.T.dot(probs - y_batch) / self.batch_size
                self.w -= self.alpha * grad

                # Compute loss
                epsilon = 1e-7
                loss = -np.sum(y_batch * np.log(probs + epsilon)
                               ) / self.batch_size
                total_loss += loss

            # Compute average loss and loss change
            avg_loss = total_loss / (n_samples // self.batch_size)
            loss_change = (previous_loss - avg_loss) / previous_loss

            # Check the direction of loss change and adjust alpha accordingly
            if loss_change < 0:  # Loss is increasing
                self.alpha *= .5
                print(
                    f"Epoch {epoch+1}: Decreasing alpha to {self.alpha} due to increased loss.")
            elif loss_change < loss_threshold:  # Loss decrease is below the threshold
                self.alpha *= 1.2  # Increase alpha by 10%
                print(
                    f"Epoch {epoch+1}: Increasing alpha to {self.alpha} due to slow decrease in loss.")

            previous_loss = avg_loss
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss}")

            last_sample_prediction = np.argmax(probs[1])
            last_sample_actual = np.argmax(y_batch[1])
            print(
                f"Last sample prediction: {last_sample_prediction}, Actual: {last_sample_actual}")

        # Plot confusion matrix after training
        self.plot_confusion_matrix(X, y)

    # Method to predict the class of a given input
    def predict(self, X: np.ndarray):
        z = X.dot(self.w)
        probs = self.softmax(z)
        y_hat = np.argmax(probs, axis=1).reshape(-1, 1)
        return y_hat
