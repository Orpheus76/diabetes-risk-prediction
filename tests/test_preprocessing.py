import numpy as np
import pandas as pd
from src.preprocessing import split_data, impute_missing_values, COLS_TO_IMPUTE


df = pd.read_csv("data/raw/diabetes.csv")


def test_split_proportions():
    X_train, X_test, y_train, y_test = split_data(df)
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

    assert abs(len(X_test) / len(df) - 0.2) < 0.01
    assert abs(len(X_train) / len(df) - 0.8) < 0.01
    assert abs(len(y_test) / len(df) - 0.2) < 0.01
    assert abs(len(y_train) / len(df) - 0.8) < 0.01


def test_random_state_reproducibility():
    X_train1, X_test1, y_train1, y_test1 = split_data(df, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(df, random_state=42)
    
    assert X_train1.equals(X_train2)
    assert X_test1.equals(X_test2)
    assert y_train1.equals(y_train2)
    assert y_test1.equals(y_test2)


def test_random_state_different_gives_different_split():
    X_train1, X_test1, y_train1, y_test1 = split_data(df, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(df, random_state=0)
    
    assert not X_train1.equals(X_train2)
    assert not X_test1.equals(X_test2)
    assert not y_train1.equals(y_train2)
    assert not y_test1.equals(y_test2)


def test_imputation_replaces_zeros_with_median():
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test)

    for c in COLS_TO_IMPUTE:
        assert (X_train_imputed[c] == 0).sum() == 0
        assert (X_test_imputed[c] == 0).sum() == 0

        median = X_train[c].replace(0, np.nan).median()
        assert (X_train_imputed[c] == median).sum() > 0
        assert (X_test_imputed[c] == median).sum() > 0