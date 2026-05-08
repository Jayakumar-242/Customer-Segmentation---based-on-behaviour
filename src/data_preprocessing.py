import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"]


def load_data(path: str = "data/customers.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Gender_Enc"] = (df["Gender"] == "Female").astype(int)
    region_dummies = pd.get_dummies(df["Region"], prefix="Region", drop_first=True)
    df = pd.concat([df, region_dummies], axis=1)
    return df


def scale_features(df: pd.DataFrame, features: list = None) -> tuple[pd.DataFrame, StandardScaler]:
    features = features or NUMERIC_FEATURES
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled, scaler


def preprocess(path: str = "data/customers.csv") -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    df_raw = load_data(path)
    df_clean = clean_data(df_raw)
    df_encoded = encode_categoricals(df_clean)
    df_scaled, scaler = scale_features(df_encoded)
    return df_raw, df_scaled, scaler
