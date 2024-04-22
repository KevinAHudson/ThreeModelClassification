
from sklearn.preprocessing import StandardScaler
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class ClassificationModel(BaseModel):
    """
        Abstract class for classification models
    """

    # check if the matrix is 2-dimensional. if not, raise an exception
    def _check_matrix(self, mat, name):
        if len(mat.shape) != 2:
            raise ValueError(
                f"Your matrix {name} shape is not 2D! Matrix {name} has the shape {mat.shape}")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            train classification model
            Args:
                X:  Input data
                y:  targets/labels
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """
            apply the learned model to input X
            parameters
            ----------
            X     2d array
                  input data
        """
        pass
