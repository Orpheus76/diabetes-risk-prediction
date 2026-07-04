import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# La liste des colonnes qui contiennent des valeurs manquantes et qui nécessitent une imputation
COLS_TO_IMPUTE = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def split_data(df, target_col='Outcome', test_size=0.2, random_state=42):
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test


def impute_missing_values(X_train, X_test, cols=COLS_TO_IMPUTE):
    
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    for c in cols:
        X_train_clean[c] = X_train_clean[c].replace(0, np.nan)
        X_test_clean[c] = X_test_clean[c].replace(0, np.nan)

    medians = X_train_clean[cols].median()
    X_train_clean[cols] = X_train_clean[cols].fillna(medians)
    X_test_clean[cols] = X_test_clean[cols].fillna(medians)

    return X_train_clean, X_test_clean