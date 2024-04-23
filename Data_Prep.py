from abc import ABC, abstractmethod
from util.data import binarize_classes, dataframe_to_array
from datasets.MNISTDataset import MNISTDataset
import matplotlib.pyplot as plt
import numpy as np


class DataPreparation():
    def __init__(self, target_pipe, feature_pipe):
        self.target_pipe = target_pipe
        self.feature_pipe = feature_pipe

    @abstractmethod
    def data_prep(self):
        pass

    def fit(self, X, y=None):
        if self.target_pipe is not None:
            self.target_pipe.fit(y)

        if self.feature_pipe is not None:
            self.feature_pipe.fit(X)

    def transform(self, X, y=None):
        if self.target_pipe is not None:
            y = self.target_pipe.transform(y)

        if self.feature_pipe is not None:
            X = self.feature_pipe.transform(X)

        return X, y

    def fit_transform(self, X, y):
        self.fit(X, y)
        X, y = self.transform(X, y)
        return X, y


class MNISTData_Prep(DataPreparation):
    def __init__(self, target_pipe, feature_pipe):
        super().__init__(target_pipe, feature_pipe)

    def plot_distribution(self, y, title):
        unique, counts = np.unique(y, return_counts=True)
        plt.bar(unique, counts)
        plt.title(title)
        plt.xlabel('Labels')
        plt.ylabel('Occurrences')
        plt.show()

    def data_prep(self, binarize=False, return_array=False):
        mnist_dataset = MNISTDataset()
        X_trn_df, y_trn_df, X_vld_df, y_vld_df = mnist_dataset.load()

        # Plot distribution before binarization
       # self.plot_distribution(y_trn_df, 'Before Binarization')

        # Converts MNIST problem to classifying ONLY 1s vs 0s
        if binarize:
            X_trn_df, y_trn_df = binarize_classes(
                X_trn_df,
                y_trn_df,
                pos_class=[1],
                neg_class=[0],
            )
            X_vld_df, y_vld_df = binarize_classes(
                X_vld_df,
                y_vld_df,
                pos_class=[1],
                neg_class=[0],
            )

            # Plot distribution after binarization
           # self.plot_distribution(y_trn_df, 'After Binarization')

        X_trn_df, y_trn_df = self.fit_transform(X=X_trn_df, y=y_trn_df)
        X_vld_df, y_vld_df = self.transform(X=X_vld_df, y=y_vld_df)

        if return_array:
            print("Returning data as NumPy array...")
            return dataframe_to_array([X_trn_df, y_trn_df, X_vld_df, y_vld_df])

        return X_trn_df, y_trn_df, X_vld_df, y_vld_df
