# MSI-projekt
Projekt pod tytułem: "Implementacja algorytmu uczenia zespołowego Random Stumps, gdzie pojedynczą składową jest pień decyzyjny (drzweo decyzyjne o jednej gałęzi - stump)"

W pliku 'real_data.csv' znajdują się dane rzeczywiste "Bank Customer Churn Dataset", pobrany z portaflu Keggle.

W folderze "Program" znajdują się kody programu naisanych w Pythonie. Program 'main.py' jest programem to wystartowania i spowoduje działanie pozostałych programów. 
Wszystkie eksperymenty i tworzenie wykresów robią się w pliku 'experiments.py', aby wyniki były powtarzalne powstał 'random_state.py', w którym jest zainicjonowany random_state. 
Algorytm Random Stumps składa się z dwóch programów - pierwszy tworzy drzewa decyzyjne o jednym poziomie decyzji - 'decision_stump.py' oraz drugiego 'random_stumps.py' - tworzy on Random Forrest, z tych drzew dyzcyjnych o jednym poziomie decyzji. 
