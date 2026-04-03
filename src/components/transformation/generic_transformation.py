import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from src.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.config_entity import GenericDataTransformationConfigEntity
from src.entity.artifact_entity import DataValidationArtifact, GenericDataTransformationArtifact
from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.utils.utils import write_npy_file
from src.components.processing.custom_transformer import ColumnDatatypeTransformer
from src.components.transformation.base_data_transformation import BaseDataTransformation
        
class GenericDataTransformation(BaseDataTransformation):
    def __init__(
            self, 
            data_transformation_config: GenericDataTransformationConfigEntity,
            data_validation_artifact: DataValidationArtifact,
    ):
            super().__init__(data_transformation_config, data_validation_artifact)

    def pipeline_init(
            self
    ) -> Pipeline:
        try:
            logging.info("Preparing data transformation pipeline")
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            pipeline = Pipeline(steps=[
                ('imputer', imputer),
                ('column_datatype_transformer', ColumnDatatypeTransformer()),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            return pipeline
        except Exception as e:
            logging.error(f"Error occurred in initiate_pipeline: {e}")
            raise NetException(e, sys) from e
        
    def save_transformed_data(
            self,
            transformed_x_train_df: pd.DataFrame,
            transformed_x_test_df: pd.DataFrame,
            y_train_series: pd.Series,
            y_test_series: pd.Series
    ) -> tuple[str, str]:
        try:
            logging.info("Saving transformed data...")
            train_concatenated_df = pd.concat([transformed_x_train_df, y_train_series], axis=1)
            test_concatenated_df = pd.concat([transformed_x_test_df, y_test_series], axis=1)

            train_concatenated_arr = train_concatenated_df.values
            test_concatenated_arr = test_concatenated_df.values

            generic_transformed_train_file_path = self.data_transformation_config.generic_transformed_train_file_path
            generic_transformed_test_file_path = self.data_transformation_config.generic_transformed_test_file_path

            write_npy_file(generic_transformed_train_file_path, train_concatenated_arr)
            write_npy_file(generic_transformed_test_file_path, test_concatenated_arr)

            return generic_transformed_train_file_path, generic_transformed_test_file_path
        except Exception as e:
            logging.error(f"Error occurred in save_transformed_data: {e}")
            raise NetException(e, sys) from e
        
    def wrap_artifact(
            self,
            train_file_path: str,
            test_file_path: str,
            object_file_path: str
    ) -> GenericDataTransformationArtifact:
        try:
            logging.info("Wrapping data transformation artifact...")

            data_transformation_artifact = GenericDataTransformationArtifact(
                transformed_train_file_path=train_file_path,
                transformed_test_file_path=test_file_path,
                transformed_object_file_path=object_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occurred in wrap_artifact: {e}")
            raise NetException(e, sys) from e