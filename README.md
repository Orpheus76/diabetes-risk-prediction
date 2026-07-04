# Diabetes Risk Prediction

Prédiction du risque de diabète à partir de données médicales issues du **Pima Indians Diabetes Dataset**, avec comparaison de plusieurs modèles et interprétabilité via SHAP.

## Objectif

Prédire si un patient présente un risque de diabète à partir de 8 mesures cliniques (glucose, pression artérielle, IMC, âge, etc.), puis identifier les variables qui influencent le plus la prédiction.

## Dataset

- Source : [Pima Indians Diabetes Database](https://archive.ics.uci.edu/dataset/34/diabetes)
- 768 observations, 8 variables cliniques et 1 variable cible binaire (`Outcome`)
- Problème principal : certaines colonnes contiennent des zéros médicalement peu plausibles (par exemple `Insulin = 0`), qui sont traités comme des valeurs manquantes

## Méthodologie

1. **Exploration des données (EDA)** : analyse des distributions, des corrélations et détection des valeurs manquantes déguisées en zéros.
2. **Prétraitement** : séparation train/test stratifiée, puis imputation des valeurs manquantes à l’aide de la médiane calculée uniquement sur l’ensemble d’entraînement afin d’éviter le data leakage.
3. **Modélisation** : entraînement d’une régression logistique comme baseline interprétable, puis d’une forêt aléatoire.
4. **Évaluation** : comparaison des modèles avec les métriques accuracy, precision, recall, F1-score, ROC-AUC et matrice de confusion ; le recall est particulièrement important dans un contexte médical.
5. **Interprétabilité** : utilisation de SHAP pour analyser l’importance des variables et mieux comprendre les prédictions.

## Résultats

_Résultats à compléter après l’entraînement des modèles._

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression |  |  |  |  |  |
| Random Forest |  |  |  |  |  |

## Installation et exécution

```bash
git clone https://github.com/Orpheus76/Diabetes-Risk-Prediction.git
cd diabetes-risk-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data_loader.py
jupyter notebook notebooks/01_eda.ipynb
```

## Structure du projet

```text
diabetes-risk-prediction/
├── data/raw/            # Données brutes
├── notebooks/           # Exploration et analyse
├── src/                 # Code réutilisable (chargement, prétraitement, entraînement)
├── models/              # Modèles entraînés
└── reports/figures/     # Visualisations exportées
```

## Pistes d’amélioration

- Tester d’autres modèles, comme XGBoost ou LightGBM
- Mieux gérer le déséquilibre des classes, par exemple avec SMOTE
- Déployer le modèle via une API légère avec FastAPI ou Flask
- Ajouter une validation croisée pour obtenir une évaluation plus robuste
