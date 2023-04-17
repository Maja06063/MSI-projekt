from sklearn.model_selection import train_test_split
from random_stumps import RandomStumps
from data_initializer import DataInitializer
from sklearn.metrics import accuracy_score
from ref_methods import RefMethods
from show_results import ShowResults
from random_state import RANDOM_STATE
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

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
            data_for_plots = data_input # zmienna do warunku zeby wykresy tylko do syntetycznych robic
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

        # Implementacja funckji do podziału za pomocą startyfikowanej wielokttonej walidacji krzyżowej 5:2
        splits = 5
        repeats = 2
        iter = range(1,11)
        rskf = RepeatedStratifiedKFold(n_splits= splits, n_repeats= repeats, random_state= RANDOM_STATE)
        

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
                scores = []
                # Podział na zbiory treningowe i testowe - walidacja krzyżowa 5:2 i przeprowadzenie testu
                for train_index, test_index in rskf.split(data_x, data_y):
                    x_train, x_test = data_x[train_index], data_x[test_index]
                    y_train, y_test = data_y[train_index], data_y[test_index]
                    n_classes = max(y_train) + 1
                    random_stumps_algorithm = RandomStumps(n_classes)
                    random_stumps_algorithm.fit(x_train,y_train)
                    y_predict = random_stumps_algorithm.predict(x_test)
                    scores.append(accuracy_score(y_test, y_predict))
                
                # Wyswietlanie prcesu nauki dla kazdego folda
                fig0 = plt.figure('Figure 0')
                plt.plot(iter, scores)
                plt.title('Random Stumps metryka a foldy')
                plt.savefig('../RS_foldy')
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print("Metryka dokladnosci klasyfikacji Random Stumps wynosi: %.3f (%.3f)" % (mean_score, std_score))
                
                # Wyswietlanie subplotow dla zbiory syntetycznego dla prawdziwej etykiety (po lewej) i predykcji (po prawej)
                if data_for_plots == "1":
                    fig1 = plt.figure('Figure 1')
                    fig, ax = plt.subplots(1,2)
                    fig.suptitle("Random Stumps")
                    ax[0].scatter(x_test[:, 0], x_test[:, 1], c= y_test, cmap= 'bwr')
                    ax[1].scatter(x_test[:, 0], x_test[:, 1], c= y_predict, cmap= 'bwr')
                    plt.savefig('../Random_Stumps.png') 

                self.show_results.random_stumps_score = mean_score

            # Boosting
            if "2" in data_input:
                algorithms_loop = False
                scores = []
                print("Uruchomiono algorytm boosting")
                for train_index, test_index in rskf.split(data_x, data_y):
                    x_train, x_test = data_x[train_index], data_x[test_index]
                    y_train, y_test = data_y[train_index], data_y[test_index]
                    boosting_algorithm = self.ref_methods.boosting
                    boosting_algorithm.fit(x_train,y_train)
                    y_predict = boosting_algorithm.predict(x_test)
                    scores.append(accuracy_score(y_test, y_predict))

                # Wyswietlanie prcesu nauki dlakazdego folda
                fig2 = plt.figure('Figure 2')
                plt.plot(iter, scores)
                plt.title('Boosting metryka a foldy')
                plt.savefig('../Boost_foldy')
                #plt.show()
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print("Metryka klasyfikacji Boosting wynosi: %.3f (%.3f)" % (mean_score, std_score))

                # Wyswietlanie subplotow dla zbiory syntetycznego dla prawdziwej etykiety (po lewej) i predykcji (po prawej)
                if data_for_plots == "1":
                    fig3 = plt.figure('Figure 3')
                    fig, ax = plt.subplots(1,2)
                    fig.suptitle("Boosting")
                    ax[0].scatter(x_test[:, 0], x_test[:, 1], c= y_test, cmap= 'bwr')
                    ax[1].scatter(x_test[:, 0], x_test[:, 1], c= y_predict, cmap= 'bwr')
                    plt.savefig('../Boosting.png')

                self.show_results.boosting_score = mean_score

            # Bagging:
            if "3" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm bagging")
                scores = []
                for train_index, test_index in rskf.split(data_x, data_y):
                    x_train, x_test = data_x[train_index], data_x[test_index]
                    y_train, y_test = data_y[train_index], data_y[test_index]
                    bagging_algorithm = self.ref_methods.bagging
                    bagging_algorithm.fit(x_train,y_train)
                    y_predict = bagging_algorithm.predict(x_test)
                    scores.append(accuracy_score(y_test, y_predict))

                # Wyswietlanie prcesu nauki dla kazdego folda
                fig4 = plt.figure('Figure 4')
                plt.plot(iter, scores)
                plt.title('Bagging metryka a foldy')
                plt.savefig('../Bagg_foldy')

                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print("Metryka klasyfikacji Bagging wynosi: %.3f (%.3f)" % (mean_score, std_score))
                
                # Wyswietlanie subplotow dla zbiory syntetycznego dla prawdziwej etykiety (po lewej) i predykcji (po prawej)
                if data_for_plots == "1":
                    fig5 = plt.figure('Figure 5')
                    fig, ax = plt.subplots(1,2)
                    fig.suptitle("Bagging")
                    ax[0].scatter(x_test[:, 0], x_test[:, 1], c= y_test, cmap= 'bwr')
                    ax[1].scatter(x_test[:, 0], x_test[:, 1], c= y_predict, cmap= 'bwr')
                    plt.savefig('../Bagging.png')
                
                self.show_results.bagging_score = mean_score

            # Regresja logistyczna:
            if "4" in data_input:
                algorithms_loop = False
                print("Uruchomiono algorytm regresji logistycznej")
                scores = []
                for train_index, test_index in rskf.split(data_x, data_y):
                    x_train, x_test = data_x[train_index], data_x[test_index]
                    y_train, y_test = data_y[train_index], data_y[test_index]
                    log_reg_algorithm = self.ref_methods.logistic_regression
                    log_reg_algorithm.fit(x_train,y_train)
                    y_predict = log_reg_algorithm.predict(x_test)
                    scores.append(accuracy_score(y_test, y_predict))

                # Wyswietlanie prcesu nauki dlakazdego folda
                fig6 = plt.figure('Figure 6')
                plt.plot(iter, scores)
                plt.title('Regresja logistyczna metryka a foldy')
                plt.savefig('../RL_foldy')

                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print("Metryka klasyfikacji regresji logistycznej wynosi: %.3f (%.3f)" % (mean_score, std_score))
                
                # Wyswietlanie subplotow dla zbiory syntetycznego dla prawdziwej etykiety (po lewej) i predykcji (po prawej)
                if data_for_plots == "1":
                    fig7 = plt.figure('Figure 7')
                    fig, ax = plt.subplots(1,2)
                    fig.suptitle("Regresja logistyczna")
                    ax[0].scatter(x_test[:, 0], x_test[:, 1], c= y_test, cmap= 'bwr')
                    ax[1].scatter(x_test[:, 0], x_test[:, 1], c= y_predict, cmap= 'bwr')
                    plt.savefig('../Regresja_logistyczna.png')
                
                self.show_results.logistic_regression_score = mean_score

        self.show_results.write_results_to_file('../output_file')