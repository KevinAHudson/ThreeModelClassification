import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from BaseModel import ClassificationModel
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the Naive Bayes class, which inherits from ClassificationModel


class NaiveBayes(ClassificationModel):
    # Initialize the class with smoothing parameter
    def __init__(self, smoothing: float = 1e-2):
        ClassificationModel.__init__(self)
        self.smoothing = smoothing
        self.class_labels = None
        self.priors = None
        self.log_priors = None
        self.means = None
        self.stds = None

    # Method to compute log of Gaussian distribution
    def log_gaussian_distribution(self, X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        return norm.logpdf(X, mu, std**2)

    # Method to compute priors (class probabilities)
    def compute_priors(self, y: np.ndarray) -> None:
        unique_labels, counts = np.unique(y, return_counts=True)
        self.priors = counts / y.shape[0]
        self.log_priors = np.log(self.priors)

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
            '/Users/kevinhudson/Documents/Classification Mini Project/confusion_matrix_naive_bayes.png')
        plt.show()

    # Method to compute features
    def compute_features(self, X: np.ndarray, y: np.ndarray) -> None:
        # Calculate means and std for each feature per class
        self.class_labels = np.unique(y)
        label_indices = [np.where(y == label)[0]
                         for label in self.class_labels]
        self.means = np.array([X[label_indices[i]].mean(axis=0)
                              for i in range(len(self.class_labels))])
        self.stds = np.array([X[label_indices[i]].std(axis=0)
                             for i in range(len(self.class_labels))])

        # Apply smoothing to std
        self.stds += self.smoothing

    def compute_log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        # Calculate likelihoods per class
        log_likelihoods = np.array([self.log_gaussian_distribution(
            X, mu, std) for mu, std in zip(self.means, self.stds)])
        return log_likelihoods.sum(axis=2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Training
        self.compute_features(X, y)
        self.compute_priors(y)

        self.plot_confusion_matrix(X, y)

    def predict(self, X) -> np.ndarray:
        # Make predictions based on the training
        log_likelihoods = self.compute_log_likelihoods(X)
        log_posteriors = log_likelihoods + self.log_priors[:, np.newaxis]
        y_hat = self.class_labels[np.argmax(log_posteriors, axis=0)]
        return y_hat.reshape(-1, 1)
