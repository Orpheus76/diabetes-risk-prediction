import os
import pandas as pd


RAW_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

# Le fichier ne contient pas d'en-têtes, donc on les définit manuellement
COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

RAW_PATH = os.path.join("data", "raw", "diabetes.csv")

def download_raw_data(save_path=RAW_PATH):
    
    df = pd.read_csv(RAW_URL, header=None, names=COLUMN_NAMES)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def load_raw_data(path=RAW_PATH):

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = download_raw_data(save_path=path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df.shape)