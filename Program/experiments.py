from bagging  import Bagging
from boosting import Boosting
from logistic_regression import LogisticRegression
from random_stumps import RandomStumps
from data_initializer import DataInitializer

class Experiments():

    data_init = DataInitializer ()
    def __init__(self):
        pass

    #metoda przeprowadzająca eksperymenty
    def run(self):
        while True:
            print("1.Symetryczne\n2.Realne\n3.Oba\n4.Wyjscie\n")
            data_input = input("Które dane neleży użyć? Wciśnij 1, 2 lub 3\n")
            if data_input == "1":
                self.data_init.prepare_artificial_data()
                break
            elif data_input == "2":
                self.data_init.prepare_real_data("")
                break
            elif data_input == "3":
                self.data_init.prepare_artificial_data()
                self.data_init.prepare_real_data("")
                break
            elif data_input == "4":
                exit(0)
            else:
                print("Niepoprawny wybor!Sprobuj jeszcze raz\n")
        data_x = self.data_init.get_x_data_copy()
        data_y = self.data_init.get_y_data_copy()

        algorithms_loop = True
        while algorithms_loop:
            print("1.Random Stumps\n2.boosting\n3.bagging\n4.Regresja Logistyczna\n5.Wyjscie z programu\n")
            data_input = input("Ktory algorytm uruchomic?Podaj numery algorytmow, bez spacji!\nPrzykład:13\n")

            if "5" in data_input:
                exit(0)

            if "1" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm Random Stumps")
                # TODO implementacja podziału na zbiór uczący i treningowy, oraz wywołanie fita, predicta itp.

            if "2" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm boosting")
                # TODO implementacja podziału na zbiór uczący i treningowy, oraz wywołanie fita, predicta itp.

            if "3" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm bagging")
                # TODO implementacja podziału na zbiór uczący i treningowy, oraz wywołanie fita, predicta itp.

            if "4" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm regresji logistycznej")
                # TODO implementacja podziału na zbiór uczący i treningowy, oraz wywołanie fita, predicta itp.