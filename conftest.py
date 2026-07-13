import pytest
from src.data_loader import load_raw_data
from src.preprocessing import split_data, impute_missing_values, scale_features


@pytest.fixture(scope="session")

def raw_data():

    """
    Fournit le DataFrame brut (diabetes.csv) en mémoire pour l'ensemble des tests.
    Utilise load_raw_data() qui gère automatiquement le chemin vers data/raw/diabetes.csv
    """
    return load_raw_data()


@pytest.fixture(scope="session")

def preprocessed_data(raw_data):

    """
    Génère en mémoire les données nettoyées, imputées et standardisées pour les tests d'entraînement.
    Evite toute dépendance envers des fichiers sur disque (data/processed/) qui sont ignorés par Git (.gitignore)
    """

    X_train, X_test, y_train, y_test = split_data(raw_data)
    X_train_clean, X_test_clean = impute_missing_values(X_train, X_test)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train_clean, X_test_clean)

    return X_train_scaled, X_test_scaled, y_train.values.ravel(), y_test.values.ravel()    
    
