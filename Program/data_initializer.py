from sklearn import datasets

class DataInitializer():
#metody, które mają na celu przygotować dane do eksperymentów
    
    
    data_x= [] #dwuwymiarowa tablica cech
    data_y = [] #jednowymiarowa tablica poprawnych odpowiedzi

    def prepare_artificial_data(self):

        self.data_x, self.data_y = datasets.make_classification(
            n_samples= 500
        )
    #input_file - plik z ktorego bedziemy brac realne
    def prepare_real_data(self, input_file: str):
        pass 
    
    def get_x_data_copy(self):
        return self.data_x.copy()
    
    def get_y_data_copy(self):
        return self.data_y.copy()

    