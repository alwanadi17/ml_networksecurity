import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

from src.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.config_entity import CatBoostDataTransformationConfigEntity
from src.entity.artifact_entity import DataValidationArtifact, CatBoostDataTransformationArtifact
from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.utils.utils import write_parquet_file, write_yaml_file
from src.components.processing.custom_transformer import ColumnDatatypeTransformer
from src.components.transformation.base_data_transformation import BaseDataTransformation
        
class CatBoostDataTransformation(BaseDataTransformation):
    def __init__(
            self,
            data_transformation_config: CatBoostDataTransformationConfigEntity,
            data_validation_artifact: DataValidationArtifact
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
                ('column_datatype_transformer', ColumnDatatypeTransformer())
            ])

            return pipeline
        except Exception as e:
            logging.error(f"Error occurred in initiate_pipeline: {e}")
            raise NetException(e, sys) from e
        
    def save_data_config(
            self,
            transformed_x_train_df: pd.DataFrame
    ) -> None:
        try:
            cat_features = pd.DataFrame(transformed_x_train_df).columns
            data_config = {
                'cat_features': list(cat_features)
            }
            data_config_file_path = self.data_transformation_config.data_config_file_path
            write_yaml_file(data_config_file_path, data_config)
        except Exception as e:
            logging.error(f"Error occurred in save_data_config: {e}")
            raise NetException(e, sys) from e
        
    def save_transformed_data(
            self,
            transformed_x_train_df: pd.DataFrame,
            transformed_x_test_df: pd.DataFrame,
            y_train_series: pd.Series,
            y_test_series: pd.Series
    ) -> tuple[str, str]:
        try:
            logging.info("Saving data config...")
            self.save_data_config(transformed_x_train_df)

            logging.info("Saving transformed data...")
            train_concatenated_df = pd.concat([transformed_x_train_df, y_train_series], axis=1)
            test_concatenated_df = pd.concat([transformed_x_test_df, y_test_series], axis=1)

            transformed_train_file_path = self.data_transformation_config.catboost_transformed_train_file_path
            transformed_test_file_path = self.data_transformation_config.catboost_transformed_test_file_path

            write_parquet_file(transformed_train_file_path, train_concatenated_df)
            write_parquet_file(transformed_test_file_path, test_concatenated_df)

            return transformed_train_file_path, transformed_test_file_path
        except Exception as e:
            logging.error(f"Error occurred in save_transformed_data: {e}")
            raise NetException(e, sys) from e
        
    def wrap_artifact(
            self,
            train_file_path: str,
            test_file_path: str,
            object_file_path: str
    ) -> CatBoostDataTransformationArtifact:
        try:
            logging.info("Wrapping data transformation artifact...")

            data_config_file_path = self.data_transformation_config.data_config_file_path

            data_transformation_artifact = CatBoostDataTransformationArtifact(
                transformed_train_file_path=train_file_path,
                transformed_test_file_path=test_file_path,
                transformed_object_file_path=object_file_path,
                data_config_file_path=data_config_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occurred in wrap_artifact: {e}")
            raise NetException(e, sys) from e