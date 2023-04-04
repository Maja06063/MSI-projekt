from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

"""
Klasa RefMethods służy do przechowywania metod referencyjnych.
"""

class RefMethods():
    bagging = BaggingClassifier ()
    boosting = GradientBoostingClassifier ()
    logistic_regression = LogisticRegression ()
