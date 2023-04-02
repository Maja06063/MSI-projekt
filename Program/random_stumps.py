from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np

class RandomStumps (BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        #Dla Oli: Uzupełnij metodę
        pass

    def fit(self,x_train,y_train):
        #Dla Oli: Uzupełnij metodę
        print("Tu bedzie fit")
        return self

    def predict(self,x_test):
        #Dla Oli: Uzupełnij metodę
        print("Tu bedzie predict")
        #place_holder tymczasowo zwraca same 0, aby program nie wywalał się, bo jeszcze nie ma 
        #napisanego predicta
        place_holder = [0] * len(x_test)
        return place_holder