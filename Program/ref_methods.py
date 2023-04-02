from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

class RefMethods():
    bagging = BaggingClassifier ()
    boosting = GradientBoostingClassifier ()
    logistic_regression = LogisticRegression ()
