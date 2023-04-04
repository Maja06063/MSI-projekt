from sklearn import datasets

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
            n_samples= 500
        )

    """
    Metoda prepare_real_data służy do przygotowania danych tabelarycznych z pliku.
    Argumenty:
    1. input_file = plik z ktorego bedziemy brac realne
    """
    def prepare_real_data(self, input_file: str):
        pass

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


    
