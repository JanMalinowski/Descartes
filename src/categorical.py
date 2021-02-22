import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from typing import List


class Encoder:
    """
    A simple wrapper for label encoder for encoding multiple columns.
    """

    def __init__(self, cols: List[str]) -> None:
        self.cols = cols
        self.encoders = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.cols:
            le = preprocessing.LabelEncoder()
            le.fit(df[col].values)
            self.encoders[col] = le

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            df[col] = self.encoders[col].transform(df[col])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str) -> None:
        joblib.dump(self.encoders, path)

    def load(self, path: str) -> None:
        self.encoders = joblib.load(path)
