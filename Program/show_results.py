
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
    def show_graphs(self):
        pass
