from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from random_state import RANDOM_STATE

"""
Klasa RefMethods służy do przechowywania metod referencyjnych.
"""

class RefMethods():
    bagging = BaggingClassifier (random_state = RANDOM_STATE)
    boosting = GradientBoostingClassifier (random_state = RANDOM_STATE)
    logistic_regression = LogisticRegression (random_state= RANDOM_STATE)
