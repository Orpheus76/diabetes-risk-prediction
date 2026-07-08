import pandas as pd
from src.train import train_logistic_regression, train_random_forest, evaluate_model

# Chargement des données nettoyées que nous avons créées précédemment

X_train = pd.read_csv("data/processed/X_train_clean.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# .values.ravel() sert à transformer le DataFrame de y_train en un tableau numpy 1D 
# (indispensable pour que le modèle puisse l'utiliser correctement)

X_test = pd.read_csv("data/processed/X_test_clean.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()


def test_training_models():

    # 1. Entraîner la régression logistique
    lr_model = train_logistic_regression(X_train, y_train)
    
    assert hasattr(lr_model, 'predict')
    assert hasattr(lr_model, 'predict_proba')

    # 2. Entraîner la forêt aléatoire
    rf_model = train_random_forest(X_train, y_train)

    assert hasattr(rf_model, 'predict')
    assert hasattr(rf_model, 'predict_proba')

    
def test_evaluate_model():

    lr_model = train_logistic_regression(X_train, y_train)
    scores = evaluate_model(lr_model, X_test, y_test)

    # 1. Vérifier que le résultat est bien un dictionnaire (isinstance(scores, dict))
    assert isinstance(scores, dict)

    # 2. Vérifier que les 5 clés sont bien présentes
    expected_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    for key in expected_keys:
        assert key in scores
        assert 0 <= scores[key] <= 1
