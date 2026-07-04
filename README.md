# Diabetes Risk Prediction

Prédiction du risque de diabète à partir de données médicales (Pima Indians Diabetes Dataset), avec comparaison de plusieurs modèles et interprétabilité via SHAP.

## Objectif

Prédire si un patient présente un risque de diabète à partir de 8 mesures cliniques (glucose, pression artérielle, IMC, âge, etc.), et comprendre quels facteurs pèsent le plus dans la prédiction.

## Dataset

- Source : [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 patients, 8 variables cliniques, 1 variable cible binaire (`Outcome`)
- Défi principal : plusieurs colonnes contiennent des zéros non plausibles médicalement (ex. Insulin = 0), traités comme valeurs manquantes.

## Méthodologie

1. **Exploration des données (EDA)** : distributions, corrélations, détection des valeurs manquantes déguisées en zéros
2. **Préparation** : split train/test stratifié, imputation des valeurs manquantes (médiane calculée sur le train uniquement, pour éviter le data leakage)
3. **Modélisation** : Logistic Regression (baseline interprétable) puis Random Forest
4. **Évaluation** : accuracy, precision, recall, F1, ROC-AUC, matrice de confusion (le recall est particulièrement important dans un contexte médical)
5. **Interprétabilité** : analyse SHAP pour comprendre l'importance des features

## Résultats

_A compléter une fois les modèles entraînés._

| Modèle              | Accuracy | Precision | Recall | F1  | ROC-AUC |
| ------------------- | -------- | --------- | ------ | --- | ------- |
| Logistic Regression |          |           |        |     |         |
| Random Forest       |          |           |        |     |         |

## Installation et exécution

````bash
```bash
git clone <url-de-ton-repo>
cd diabetes-risk-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data_loader.py   # télécharge les données brutes
jupyter notebook notebooks/01_eda.ipynb
````

## Structure du projet

```
diabetes-risk-prediction/
├── data/raw/            # données brutes
├── notebooks/           # exploration et analyse, dans l'ordre
├── src/                 # code réutilisable (chargement, préprocessing)
├── models/              # modèles entraînés
└── reports/figures/     # visualisations exportées
```

## Pistes d'amélioration

- Tester XGBoost / LightGBM en complément
- Gérer le déséquilibre des classes (SMOTE)
- Déployer le modèle via une petite API (FastAPI/Flask)
