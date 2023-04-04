from experiments import Experiments

"""
Program do przeprowadzania eksperymentów na różnych algorytmach sztucznej inteligencji.
Dostępne algorytmy:
- RandomStumps,
- Boosting,
- Bagging,
- Regresja logistyczna.

Autorki:
- Aleksandra Gadzińska,
- Maja Skibińska.
"""

# Aby nie można było importować maina w innych plikach
if __name__ == "__main__":
    experiments = Experiments()
    experiments.run()
