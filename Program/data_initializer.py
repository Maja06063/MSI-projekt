from sklearn import datasets
import pandas as pd
from random_state import RANDOM_STATE

"""
Klasa DataInitializer zawiera metody, które mają na celu przygotować dane do eksperymentów.
"""

class DataInitializer():

    data_x= [] #dwuwymiarowa tablica cech
    data_y = [] #jednowymiarowa tablica poprawnych odpowiedzi

    """
    Metoda prepare_artificial_data służy do przygotowania danych syntetycznych.
    """
    def prepare_artificial_data(self):

        self.data_x, self.data_y = datasets.make_classification(
            n_samples= 500,
            random_state = RANDOM_STATE
        )

    """
    Metoda prepare_real_data służy do przygotowania danych tabelarycznych z pliku.
    Argumenty:
    1. input_file = plik z ktorego bedziemy brac realne
    """
    def prepare_real_data(self, input_file: str):
        df = pd.read_csv(input_file, sep=',', header=0)
        self.data_y = df[df.columns[-1]].to_numpy()
        self.data_x = df[df.columns[0:-1]].to_numpy()

    """
    Metoda get_x_data_copy zwraca kopię tablicy cech.
    """
    def get_x_data_copy(self):
        return self.data_x.copy()

    """
    Metoda get_y_data_copy zwraca kopię tablicy poprawnych wyników klasyfikacji.
    """
    def get_y_data_copy(self):
        return self.data_y.copy()
