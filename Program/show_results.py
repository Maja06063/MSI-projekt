import numpy as np
from scipy.stats import ttest_ind
import pandas as pd

"""
Klasa ShowResults służy do generowania tabel i wykresów z wynikami klasyfikacji oraz zapisu ich do plików.
"""

class ShowResults():

    random_stumps_score = 0,
    bagging_score = 0,
    logistic_regression_score = 0,
    boosting_score = 0

    def write_results_to_file(self, output_file:str):

        file_write = open(output_file, 'wt') #otworzenie piku do zapisu
        file_write.write('Random stumps, bagging, logistic regression, boosting\n')
        file_write.write(str(self.random_stumps_score)) #zapis random stumps (średniej z metryk)
        file_write.write(',')
        file_write.write(str(self.bagging_score)) #zapis baggingu
        file_write.write(',')
        file_write.write(str(self.logistic_regression_score)) #zapis regresji liniowej
        file_write.write(',')
        file_write.write(str(self.boosting_score)) #zapis boostingu
        file_write.write("\n")
        file_write.close()

    def statistic_test(self, clfs_scores: np.array, clfs_names: list, alpha: float):

        clfs_num = clfs_scores.shape[0]

        # Stworzenie macierzy t_stat, p, better_res i is_significant:
        t_stat = np.zeros((clfs_num, clfs_num))
        p = np.zeros((clfs_num, clfs_num))
        better_res = np.zeros((clfs_num, clfs_num), dtype=bool)
        is_significant = np.zeros((clfs_num, clfs_num), dtype=bool)

        # Przeprowadzanie testu T studenta:
        for i in range(clfs_num):
            for j in range(clfs_num):
                t_stat[i, j], p[i, j] = ttest_ind(clfs_scores[i], clfs_scores[j])
        print("T statystyka:\n" + self.matrix_to_string(t_stat, clfs_names) + "\n\nWartość p:\n" + self.matrix_to_string(p, clfs_names))

        # Lepsze wyniki dla każdej pary klasyfikatorów, w których dodatni wynik testu T studenta:
        better_res[t_stat > 0] = True
        print("\nLepsze wyniki:")
        print(self.matrix_to_string(better_res, clfs_names))

        # Znaczące wyniki dla każdej pary klasyfikatorów, w których p jest mniejsze od progu (alfa):
        is_significant[p < alpha] = True
        print("\nCzy znaczące statystycznie:")
        print(self.matrix_to_string(is_significant, clfs_names))

        # Znaczące lepsze wyniki dla każdej pary klasyfikatorów, w których jest i znaczący i lepszy wynik jednocześnie (z poprzednich macierzy):
        significant_better_res = better_res * is_significant
        print("\nLepsze wyniki znaczące statystycznie:")
        print(self.matrix_to_string(significant_better_res, clfs_names))

    def matrix_to_string(self, matrix: np.array, headers: list) -> str:

        headers_better = []
        for header in headers:
            headers_better.append(header + " wypadł lepiej od:")

        df = pd.DataFrame(matrix, index=headers_better, columns=headers)

        return str(df)