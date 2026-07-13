# 🏥 Diabetes Risk Prediction & Clinical Explainability

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen)
![Status](https://img.shields.io/badge/Status-Educational%20%2F%20Proof--of--Concept-informational)

Projet d'apprentissage automatique (**Machine Learning**) orienté vers l'**interprétabilité clinique et pédagogique**.
L'objectif est de prédire le risque de diabète à partir de données médicales (Pima Indians Diabetes Dataset), de comparer un modèle linéaire (*Régression Logistique*) à un modèle non-linéaire (*Forêt Aléatoire*), et de décrypter les prédictions de la Forêt Aléatoire à l'aide de **SHAP** (*SHapley Additive exPlanations*).

> **Note de portée** : ce projet est un exercice pédagogique de bout en bout (EDA → prétraitement anti-leakage → modélisation → interprétabilité → analyse critique des hyperparamètres). Il n'est **pas** un outil de diagnostic validé cliniquement

---

## 🎯 Objectif du Projet & Enjeu Clinique : Comprendre les Métriques

Dans un projet d'apprentissage automatique appliqué à la santé, l'évaluation d'un modèle ne peut pas se résumer à son taux de bonne classification global (*Accuracy*). Il est indispensable d'adapter le choix des métriques aux réalités du diagnostic médical et aux conséquences pour les patients.

### 🔢 Vocabulaire de base : la matrice de confusion

Toutes les métriques ci-dessous se calculent à partir de quatre quantités, obtenues en comparant la prédiction du modèle à la réalité clinique (`Outcome`) :

| | Réalité : Diabétique | Réalité : Non-diabétique |
| --- | :---: | :---: |
| **Prédit : Diabétique** | Vrai Positif (`TP`) | Faux Positif (`FP`) |
| **Prédit : Non-diabétique** | Faux Négatif (`FN`) | Vrai Négatif (`TN`) |

### 📘 Définition claire des métriques d'évaluation :

1. **Accuracy (Exactitude globale)** : *« Sur l'ensemble de la population examinée (sains et malades), quelle proportion de patients a été classée correctement ? »*

   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

   - Elle mesure la performance globale du modèle sur toutes les classes.
   - ⚠️ **Limite en contexte médical** : Si une pathologie ne touche que 1 % d'une population, un modèle qui prédirait systématiquement l'absence de maladie (*zéro détection*, donc `TP = 0`) atteindrait tout de même **99 % d'Accuracy** grâce aux nombreux `TN`. Pourtant, sa valeur clinique serait nulle puisqu'il n'identifierait aucun patient malade. L'Accuracy seule est donc insuffisante face à des classes déséquilibrées.

2. **Recall (Sensibilité / Taux de détection)** : *« Parmi tous les patients qui sont réellement atteints de diabète, quelle proportion le modèle parvient-il à identifier ? »*

   $$\text{Recall} = \frac{TP}{TP + FN}$$

   - Le dénominateur (`TP + FN`) représente *tous* les patients réellement diabétiques, qu'ils aient été détectés ou non. Il évalue la capacité de l'algorithme à ne laisser passer aucun cas positif : un *Recall* élevé garantit que la grande majorité des patients malades sont repérés et pris en charge à temps.

3. **Precision (Valeur Prédictive Positive)** : *« Lorsque le modèle prédit qu'un patient est à risque de diabète, quelle est la probabilité que ce diagnostic soit exact ? »*

   $$\text{Precision} = \frac{TP}{TP + FP}$$

   - Le dénominateur (`TP + FP`) représente *toutes* les alertes du modèle, justifiées ou non. Par exemple, une *Precision* de 70 % signifie que sur 100 patients identifiés comme étant à risque par l'algorithme, 70 le sont réellement (les 30 autres constituant des diagnostics faussement positifs).

4. **F1-score (Compromis équilibré)** : *« Quel est l'équilibre global entre la capacité du modèle à détecter tous les malades (Recall) et la fiabilité de ses alertes (Precision) ? »*

   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times TP}{2 \times TP + FP + FN}$$

   - C'est la moyenne harmonique entre le *Recall* et la *Precision*. Elle pénalise fortement le déséquilibre entre les deux : un modèle très précis mais peu sensible (ou l'inverse) aura un F1 tiré vers le bas par sa métrique la plus faible. Elle constitue un indicateur de synthèse idéal lorsque les classes sont déséquilibrées et que l'on recherche un compromis clinique entre la sensibilité du diagnostic et la maîtrise des faux positifs.

5. **ROC-AUC (Capacité de discrimination)** : *« Quelle est la capacité de l'algorithme à hiérarchiser correctement le risque, en distinguant de manière fiable un patient malade d'un patient sain ? »*
   - Contrairement aux métriques précédentes, l'AUC n'est pas calculée à un seuil fixe : la courbe ROC trace le Recall (`TPR = TP/(TP+FN)`) contre le taux de faux positifs (`FPR = FP/(FP+TN)`) pour tous les seuils possibles entre 0 et 1, et l'AUC (*Area Under the Curve*) en mesure l'aire sous cette courbe. Un score de 0,50 équivaut à un tirage au sort aléatoire, tandis qu'un score de 1,0 représente une discrimination parfaite. Un score $\ge 0,80$ reflète un bon pouvoir discriminant en pratique médicale.

---

> [!IMPORTANT]
> **Le dilemme médical en situation de dépistage (Recall vs Precision)** :
> - **Le Faux Négatif (déficit de Recall)** : Un patient réellement diabétique n'est pas détecté par le modèle. Il ne reçoit pas d'information ni de prise en charge thérapeutique précoce, ce qui l'expose à des complications cliniques sévères (atteintes rénales, cardiovasculaires ou rétiniennes).
> - **Le Faux Positif (déficit de Precision)** : Un patient sain est identifié à tort comme suspect. Il devra subir une consultation médicale et des examens complémentaires (par exemple une prise de sang de contrôle) pour écarter le diagnostic.
>
> 👉 **En santé publique et lors d'une phase de dépistage, ne pas détecter un patient malade (Faux Négatif) a des conséquences de santé bien plus graves qu'une investigation complémentaire chez un patient sain (Faux Positif). La priorité clinique est donc de maximiser le RECALL (Sensibilité), quitte à tolérer un taux légèrement plus élevé de fausses alertes.**

---

## 📊 Dataset & Défi des "Zéros Masqués"

- **Source** : [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle / UCI Machine Learning Repository).
- **Volume** : 768 patientes (femmes de la communauté Pima, ≥ 21 ans), 8 variables cliniques et 1 variable cible binaire (`Outcome`: 0 = non-diabétique, 1 = diabétique).
- **Problématique majeure identifiée** : De nombreuses mesures vitales contiennent des `0` qui sont biologiquement impossibles :
  - `Glucose = 0` (5 observations)
  - `BloodPressure = 0` (35 observations)
  - `SkinThickness = 0` (227 observations)
  - `Insulin = 0` (374 observations, soit près de 49 % !)
  - `BMI = 0` (11 observations)

  *Traitement apporté* : Ces zéros ont été reconnus comme des **valeurs manquantes déguisées** (`np.nan`), puis imputés à l'aide de la **médiane** calculée **uniquement sur le jeu d'entraînement** (pour éliminer tout risque de *Data Leakage*).

---

## 🔬 Méthodologie & Pipeline Pédagogique

Le projet est divisé en 4 notebooks séquentiels :

1. **`01_eda.ipynb` (Exploration des données)** : Analyse visuelle des distributions, de l'asymétrie des classes (65 % vs 35 %) et des corrélations cliniques.
2. **`02_preprocessing.ipynb` (Prétraitement & Data Leakage)** : Découpage stratifié Train/Test (80 % / 20 %), imputation par la médiane et standardisation (`StandardScaler`) en veillant à ne jamais exposer les données de test à l'étape d'apprentissage.
3. **`03_modeling.ipynb` (Modélisation & Interprétabilité)** : Entraînement, comparaison et dissection des modèles via des visualisations 2D des frontières de décision, l'inspection d'un arbre de décision individuel et une analyse de contribution SHAP.
4. **`04_model_evolution.ipynb` (Évolution & Compromis MLOps)** : Étude empirique des performances cliniques (*Recall*, *Accuracy*) et du temps de calcul (*Training Latency*) selon le nombre d'arbres de décision (`n_estimators` de 1 à 800) et la profondeur maximale (`max_depth` de 2 à non bridée), avec une lecture critique de ses propres limites (évaluation sur un split unique plutôt qu'en validation croisée).

---

## 📈 Comparaison & Résultats des Modèles

Les deux algorithmes ont été évalués sur l'échantillon de test (`X_test`, 154 patientes dont 54 diabétiques) :

| Modèle | Accuracy | Precision | Recall (Sensibilité)  | F1-score | ROC-AUC | Temps d'entraînement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline linéaire) | `70.78 %` | `60.00 %` | `50.00 %` | `54.55 %` | `0.8130` | **`~5 ms`**  |
| **Random Forest** (`100 arbres`) | **`77.27 %`** | **`70.21 %`** | **`61.11 %`** | **`65.35 %`** | **`0.8181`** | `~110 ms` |

### 🔍 Analyse clinique et computationnelle des résultats :
- **Le saut de Recall (+11,1 %)** : Sur les 54 patientes diabétiques du jeu de test, la Régression Logistique en détecte 27 (*Recall = 50 %*), tandis que la Forêt Aléatoire en détecte 33 (*Recall = 61,1 %*), soit 6 patientes supplémentaires identifiées sur cet échantillon de 154 personnes.
- **L'amélioration du compromis clinique (F1-score à +10,8 %)** : En progressant de `54,55 %` à `65,35 %`, le F1-score montre que la Forêt Aléatoire n'améliore pas seulement la détection des malades (*Recall*), mais aussi la fiabilité de ses alertes (*Precision* passant de `60,00 %` à `70,21 %`).
- **Le compromis MLOps (Coût vs Performance)** : La Régression Logistique est environ 20 fois plus rapide (`~5 ms` contre `~110 ms`), un facteur à peser selon le contexte de déploiement (batch hospitalier vs application temps réel).
- **Capacité de discrimination (ROC-AUC)** : Avec un score `ROC-AUC ~ 0,82`, les deux modèles montrent une bonne aptitude globale à hiérarchiser les probabilités de risque, indépendamment du seuil de décision choisi.

### 📈 Évolution Dynamique du Modèle (*Rounds* & Profondeur) :
Étude menée dans `04_model_evolution.ipynb` pour comprendre l'effet des hyperparamètres du Random Forest :
- **Nombre d'arbres (1 à 800) et temps d'entraînement** : Le temps d'entraînement croît linéairement avec le nombre d'arbres (`~189 ms` à `n=100` → `~1517 ms` à `n=800`, soit un coût 8 fois supérieur). Le Recall atteint son maximum observé à `n=100` (`61,11 %`), puis se dégrade à `400` et `800` arbres (`53,70 %` et `57,41 %`) — un phénomène attribué à l'épuisement de la diversité du bootstrap sur un jeu d'entraînement compact (`614 patientes`).
- **Profondeur maximale (`max_depth`)** : Un bridage trop fort (`max_depth = 2 à 4`) entraîne un sous-apprentissage marqué (*Recall ≤ 46,30 %*), incapable de capturer les interactions entre `Glucose`, `Âge` et `BMI`. Laisser les arbres grandir sans contrainte (`max_depth = None`) donne le meilleur Recall observé sur ce jeu de test.
- **Configuration retenue** : `RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)` constitue le **meilleur compromis observé sur ce split train/test** entre performance clinique et coût computationnel — à confirmer par validation croisée avant tout usage en production (voir limites ci-dessous).

### ⚖️ Critique Scientifique & Limites du Dataset :

1. **Puissance statistique et évaluation sur split unique** : Le jeu de test ne compte que 154 patientes ; chaque patiente représente ≈ 0,65 % d'Accuracy et ≈ 1,85 % de Recall. Le choix d'hyperparamètres réalisé dans `04_model_evolution.ipynb` repose sur ce même split de test évalué à répétition, ce qui peut légèrement le sur-ajuster à cet échantillon précis. Une **K-Fold Cross-Validation**, voire une étude multicentrique sur une cohorte bien plus large, serait nécessaire avant toute certification médicale.
2. **Biais de sélection et généralisation restreinte (*Out-of-Distribution*)** : La base *Pima Indians* a été collectée exclusivement auprès de femmes autochtones âgées d'au moins 21 ans résidant en Arizona. Le modèle appris ne peut pas être déployé tel quel sur des populations européennes, asiatiques ou masculines sans risque important de biais de distribution démographique.
3. **Absence de biomarqueurs sanguins modernes** : Le diagnostic international du diabète repose aujourd'hui largement sur le dosage de l'hémoglobine glycée (*HbA1c*), non disponible dans ce jeu de données historique.
4. **Recall encore insuffisant pour un usage clinique réel** : Un Recall de 61 % signifie que ~2 patientes diabétiques sur 5 ne sont pas détectées au seuil de décision par défaut (0,50). C'est un point de départ pédagogique correct, mais loin du niveau de sensibilité (souvent >90 %) attendu d'un outil de dépistage réel — d'où la piste d'amélioration sur l'ajustement du seuil ci-dessous.

---

## 🧠 Interprétabilité du Modèle (SHAP)

1. **`Glucose` (Taux de sucre sanguin)** : De loin le facteur clinique le plus déterminant. Des valeurs élevées de glucose poussent fortement la prédiction vers la classe positive (diabète).
2. **`BMI` (Indice de Masse Corporelle)** : Le deuxième facteur de risque majeur. Un surpoids/obésité contribue nettement à l'augmentation du score de risque.
3. **`Age` & `DiabetesPedigreeFunction`** : L'âge avancé et les antécédents familiaux agissent comme des facteurs aggravants qui viennent consolider le diagnostic sur des cas limites.

---

## 🛠️ Installation & Exécution locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/Orpheus76/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# 2. Créer et activer l'environnement virtuel Python
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer Jupyter Notebook pour explorer les analyses
jupyter notebook
```

---

## 📁 Structure du projet

```text
diabetes-risk-prediction/
├── conftest.py                # Configuration des tests pytest
├── data/
│   ├── processed/             # Données nettoyées, imputées et standardisées
│   └── raw/diabetes.csv       # Dataset brut (Pima Indians)
├── models/                    # Modèles entraînés et exportés (.pkl)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── notebooks/                 # Notebooks pédagogiques et interactifs
│   ├── 01_eda.ipynb           # Analyse Exploratoire des Données
│   ├── 02_preprocessing.ipynb # Prétraitement rigoureux & Anti-leakage
│   ├── 03_modeling.ipynb      # Modélisation, Frontières de décision & SHAP
│   └── 04_model_evolution.ipynb # Évolution des performances & compromis MLOps
├── src/                       # Code source réutilisable et modularisé
│   ├── data_loader.py         # Chargement des données
│   ├── preprocessing.py       # Pipeline d'imputation et de standardisation
│   └── train.py               # Fonctions d'apprentissage et d'évaluation
└── tests/                     # Suite de tests unitaires automatisés
    ├── test_preprocessing.py
    └── test_train.py
```

---

## 🚀 Pistes d'amélioration futures

- [ ] **Validation croisée (K-Fold)** : Réévaluer le choix des hyperparamètres et la performance des modèles sur plusieurs splits pour obtenir des estimations plus robustes que celles issues d'un unique train/test split.
- [ ] **Optimisation du Recall via les seuils de probabilité** : Abaisser le seuil de décision (ex : passer de `0,50` à `0,35`) pour capturer un maximum de cas à risque, quitte à accepter davantage de faux positifs.
- [ ] **Gestion du déséquilibre des classes** : Expérimenter **SMOTE** (*Synthetic Minority Over-sampling Technique*) ou l'ajustement des poids (`class_weight='balanced'`).
- [ ] **Modèles de Gradient Boosting** : Évaluer et comparer **XGBoost** et **LightGBM**.
- [ ] **Interface Web Interactive (Streamlit)** : Développer un tableau de bord clinique interactif pour tester le modèle en temps réel.

