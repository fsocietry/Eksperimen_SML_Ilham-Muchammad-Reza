import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    print(f"Missing values dihapus: {before - len(df)} baris")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplikat dihapus: {before - len(df)} baris")
    return df


def remove_outliers(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    before = len(df)
    Q1 = df[feature_cols].quantile(0.25)
    Q3 = df[feature_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[feature_cols] < (Q1 - 1.5 * IQR)) |
             (df[feature_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[mask]
    print(f"Outlier dihapus: {before - len(df)} baris")
    return df


def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    print(f"Encoding target: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Fitur berhasil distandarisasi (StandardScaler)")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target_col = 'species'

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_outliers(df, feature_cols)
    df = encode_target(df, target_col)
    df = normalize_features(df, feature_cols)

    return df


def main():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'iris_raw.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'iris_preprocessing')
    output_path = os.path.join(output_dir, 'iris_preprocessing.csv')

    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    df_processed = preprocess(df)

    df_processed.to_csv(output_path, index=False)
    print(f"\nPreprocessing selesai. Data tersimpan di: {output_path}")
    print(f"Shape akhir: {df_processed.shape}")
    return df_processed


if __name__ == '__main__':
    main()
