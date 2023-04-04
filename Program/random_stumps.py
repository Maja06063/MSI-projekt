from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from collections import Counter
from decision_stump import DecisionStump

"""
Klasa RandomStumps jest klasyfikatorem implementowanym w zadaniu.
Posiada konstruktor oraz metodę fit i predict zgodne z innymi klasyfikatorami biblioteki sklearn.
"""

class RandomStumps (BaseEstimator, ClassifierMixin):

    """
    Konstruktor budujący obiekt klasyfikatora.
    Możliwe parametry:
    TODO!!!
    """
    def __init__(self, n_stumps=2, max_depth=1, min_samples_split=2, n_feature=None):
        self.n_stumps = n_stumps
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.stumps = []

    """
    Metoda fit, za pomocą której odbywa się całe uczenie algorytmu - budowanie lasu stumpów.
    Argumenty:
    1. x - dwuwymiarowa tablica. Pierwszy wymiar mówi o numerze próbki, drugi o numerze cechy,
    2. y - tablica zawierająca poparawne wyniki klasyfikacji cech dla danej próbki,
        na jej podstawie algorytm uczy się.
    """
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

    """
    Metoda _bootstrap_samples jest metodą pomocniczą.
    """
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    """
    Metoda _most_common_label jest metodą pomocniczą.
    Zwraca najbardziej popularną klasę.
    """
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    """
    Metoda predict służąca do przeprowadzenia klasyfikacji - gdy algorytm jest już nauczony metodą fit,
    używamy predict aby podejmował decyzję.
    Argumenty:
    1. x - dwuwymiarowa tablica. Pierwszy wymiar mówi o numerze próbki, drugi o numerze cechy,
        algorytm ma za zadanie dla każdego obiektu z tablicy przydzielić do jakiej klasy należy.
    Metoda zwraca tablicę zawierającą klasy obiektów ("wymyślone" przez algorytm).
    """
    def predict(self, X):
        predictions = np.array([stump.predict(X) for stump in self.stumps])
        stump_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in stump_preds])
        return predictions
