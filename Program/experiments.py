from sklearn.model_selection import train_test_split
from random_stumps import RandomStumps
from data_initializer import DataInitializer
from sklearn.metrics import accuracy_score
from ref_methods import RefMethods
from show_results import ShowResults

"""
Klasa Experiments służy do przeprowadzenia ekperymentów. Składa się przede wszystkim z metody run.
"""

class Experiments():

    data_init = DataInitializer ()
    ref_methods = RefMethods ()
    show_results = ShowResults ()
    def __init__(self):
        pass

    """
    Metoda run przeprowadza eksperymenty.
    """
    def run(self):

        # Wybór danych wejściowych do algorytmu:
        while True:
            print("1.Syntetyczne\n2.Realne\n3.Oba\n4.Wyjscie\n")
            data_input = input("Które dane neleży użyć? Wciśnij 1, 2 lub 3\n")
            if data_input == "1":
                self.data_init.prepare_artificial_data()
                break
            elif data_input == "2":
                self.data_init.prepare_real_data('../real_data.csv')
                break
            elif data_input == "3":
                self.data_init.prepare_artificial_data()
                self.data_init.prepare_real_data("")
                break
            elif data_input == "4":
                exit(0)
            else:
                print("Niepoprawny wybor!Sprobuj jeszcze raz\n")

        # Pobranie tych danych do metody run:
        data_x = self.data_init.get_x_data_copy()
        data_y = self.data_init.get_y_data_copy()

        # Podział na zbiór uczący i testowy. Docelowo będzie zastąpiony walidacją krzyżową:
        x_train, x_test, y_train, y_test = train_test_split(
            data_x,data_y,
            test_size = 0.2,
            random_state = 66
        )

        # Wybór algorytmu do uruchomienia:
        algorithms_loop = True
        while algorithms_loop:
            print("1.Random Stumps\n2.boosting\n3.bagging\n4.Regresja Logistyczna\n5.Wyjscie z programu\n")
            data_input = input("Ktory algorytm uruchomic?Podaj numery algorytmow, bez spacji!\nPrzykład:13\n")

            # Wyjście z programu:
            if "5" in data_input:
                exit(0)

            # Random Stumps:
            if "1" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm Random Stumps")

                n_classes = max(y_train) + 1
                random_stumps_algorithm = RandomStumps(n_classes)
                random_stumps_algorithm.fit(x_train,y_train)
                y_predict = random_stumps_algorithm.predict(x_test)

                score = accuracy_score(y_test,y_predict)
                print("Metryka klasyfikacji Random Stumps wynosi:"+str(score))
                self.show_results.random_stumps_score = score

            # Boosting
            if "2" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm boosting")

                boosting_algorithm = self.ref_methods.boosting
                boosting_algorithm.fit(x_train,y_train)
                y_predict = boosting_algorithm.predict(x_test)

                score = accuracy_score(y_test,y_predict)
                print("Metryka klasyfikacji Boosting wynosi:"+str(score))
                self.show_results.boosting_score = score

            # Bagging:
            if "3" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm bagging")

                bagging_algorithm = self.ref_methods.bagging
                bagging_algorithm.fit(x_train,y_train)
                y_predict = bagging_algorithm.predict(x_test)

                score = accuracy_score(y_test,y_predict)
                print("Metryka klasyfikacji Bagging wynosi:"+str(score))
                self.show_results.bagging_score = score

            # Regresja logistyczna:
            if "4" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm regresji logistycznej")

                log_reg_algorithm = self.ref_methods.logistic_regression
                log_reg_algorithm.fit(x_train,y_train)
                y_predict = log_reg_algorithm.predict(x_test)

                score = accuracy_score(y_test,y_predict)
                print("Metryka klasyfikacji regresji logistycznej wynosi:"+str(score))
                self.show_results.logistic_regression_score = score

        self.show_results.write_results_to_file('../output_file')