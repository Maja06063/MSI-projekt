from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
from collections import Counter

# node'y (pola decyzji - takie if'y gdzie nastepuje podzial zbioru- w pniu tylko jeden taki Node)

"""
Klasa Node definiuje gałęzie (liście) drzewa DecisionStump.
W RandomStumpie mamy pewną ilość DecisionStumpów, a każdy DecisionStump ma w sobie pewną ilość Nodów.
"""

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None): #* powoduje ze trzeba zrobic Node.value
        self.feature = feature
        self.threshold = threshold  #prog decyzyjny przy ktorym koniec decyzji o podziale
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self): #sprawdzanie czy ten node jest lisciem
        return self.value is not None

"""
Klasa DecisionStump definiuje stumpy (pojedyncze drzewa) potrzebne do algorytmu RandomStumps.
"""

class DecisionStump (BaseEstimator, ClassifierMixin):

    def __init__(self, min_samples_split=2, max_depth=1, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features #randomowosc przy podzbiorach
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features) #sprawdzanie czy nie przekrocza tych cech co mamy
        self.root = self.grow_stump(X, y)
        return self

    """
    Important!
    Tutaj jest właśnie problem, czy chcemy mieć depth=0, czy depth=1???
    """
    def grow_stump(self, X, y, depth=0):    #tworzenie drzewa, u nas drzewo jest z jedna decyzja, jeden poziom glebokosci, dlatego tez depth = 1 i tylko raz zbuduje rozgalezienia (dwa liscie)
        n_samples, n_feature = X.shape
        n_labels = len(np.unique(y))

        # sprawdzanie kryterium budowania drzewa (drzewo z jedna decyzja bedzie)
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feature_idxs = np.random.choice(n_feature, self.n_features, replace=False)

        # znalezienie najlepszy podzial na te *liscie*
        best_feature, best_thresh = self.best_split(X, y, feature_idxs)

        # create child nodes - stworzy sie tylko jakby jedno rozgalezienie (ze ten decyzyjny node i jedno na lewo i na prawo decyzja)
        left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)
        left = self.grow_stump(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.grow_stump(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)

    def best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # obliczanie zysku informacji (im wiekszy tym lepszy jest ten klasyfikator - drzewo)
                gain = self.information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def information_gain(self, y, X_column, threshold):
        # entropia rodzica
        parent_entropy = self.entropy(y)

        # tworzenie dzieci do entropii
        left_idxs, right_idxs = self.split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # obliczanie wag wartosci entropii dzieci
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # obliczanie zysku informacyjnego
        information_gain = parent_entropy - child_entropy
        return information_gain

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self.traverse_stump(x, self.root) for x in X])

    def traverse_stump(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_stump(x, node.left)
        return self.traverse_stump(x, node.right)
