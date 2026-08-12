from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTENC
import pandas as pd


class SMOTENCFrame(BaseEstimator):
    """DataFrame-preserving wrapper around imblearn's SMOTENC."""

    def __init__(self, categorical_features, random_state=None, k_neighbors=5):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        columns = X.columns
        dtypes = X.dtypes
        smotenc = SMOTENC(
            categorical_features=self.categorical_features,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )
        X_res, y_res = smotenc.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=columns)
        for col in columns:
            X_res[col] = X_res[col].astype(dtypes[col])
        return X_res, y_res
