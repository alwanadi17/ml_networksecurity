import os
import sys
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging

class ColumnDatatypeTransformer(
    BaseEstimator, 
    TransformerMixin
):
    def __init__(
            self
    ):
        self.feature_names_in_ = []

    def fit(
            self, 
            X: pd.DataFrame, 
            y=None
    ):
        self.feature_names_in_ = X.columns
        return self
    
    def transform(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        try:
            X_df = X.copy()
            for col in X_df.columns:
                X_df[col] = X_df[col].astype('str')
            
            return X_df
        except Exception as e:
            logging.error(f"Error occurred in DataTypeTransformer transform method: {e}")
            raise NetException(e, sys) from e
        
    def get_feature_names_out(
            self,
            input_features=None
    ):
        feature_names = self.feature_names_in_.copy()
        return feature_names