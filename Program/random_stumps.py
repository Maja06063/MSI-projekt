from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from collections import Counter
from decision_stump import DecisionStump

class RandomStumps (BaseEstimator, ClassifierMixin):
    def __init__(self, n_stumps=2, max_depth=1, min_samples_split=2, n_feature=None):
        self.n_stumps = n_stumps
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.stumps = []
        
    def fit(self, X, y):
        self.stumps = []
        for _ in range(self.n_stumps):
            stump = DecisionStump(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            stump.fit(X_sample, y_sample)
            self.stumps.append(stump)
        return self

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([stump.predict(X) for stump in self.stumps])
        stump_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in stump_preds])
        return predictions
