"""
module for K Nearest Neighbors classifier
"""
from collections import Counter
import numpy as np

def euclidean_distance(x1, x2):
    """
    compute the euclidean distance between two vectors
    """
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    """
    K Nearest Neighbors classifier
    """
    x_data = None
    y_data = None
    k = None

    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, Y):
        """
        X: training data
        Y: training labels
        """
        self.x_data = X
        self.y_data = Y

    def predict(self, X):
        """
        X: test data
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.x_data]

        # get closest k samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_data[i] for i in k_indices]


        # get most common class label with majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    