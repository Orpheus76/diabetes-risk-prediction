#  Écrire la fonction train_logistic_regression (modèle linéaire de base).
#  Écrire la fonction train_random_forest (modèle ensembliste non-linéaire).
#  Écrire la fonction evaluate_model pour calculer les métriques médicales cruciales (Recall, Precision, F1, Accuracy, ROC-AUC).
#  Créer le fichier tests/test_train.py pour tester automatiquement le bon fonction

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def train_logistic_regression(X_train, y_train, random_state=42):

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    
    # 1. Prédictions des classes 0 ou 1
    y_pred = model.predict(X_test)

    # 2. Prédictions des probabilités de la classe 1 (diabètes) pour le ROC-AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Calcul et retour de toutes les métriques dans un dictionnaire
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }