# K Nearest Neighbors
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:

    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, Y):
        self.x_train = X
        self.y_train = Y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]

    def _predict(self, x):
        # compute distance

        # get closest k samples

        # get most common class label with majority vote

