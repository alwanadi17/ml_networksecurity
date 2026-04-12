from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.entity.artifact_entity import (
    DataValidationArtifact,
    GenericDataTransformationArtifact,
    CatBoostDataTransformationArtifact
)
from src.entity.config_entity import GenericDataTransformationConfigEntity, CatBoostDataTransformationConfigEntity
from src.constant.training_pipeline import TARGET_COLUMN
from src.utils.utils import save_object, read_csv_file

from sklearn import set_config
set_config(transform_output='pandas')

import os
import sys
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Union

class BaseDataTransformation:
    def __init__(
            self,
            data_transformation_config: Union[
                GenericDataTransformationConfigEntity,
                CatBoostDataTransformationConfigEntity
            ],
            data_validation_artifact: DataValidationArtifact
    ):
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact

    def pipeline_init(
            self
    ) -> Pipeline:
        raise NotImplementedError("Subclasses should implement this method")
    
    def get_train_test_df(
            self
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        try:
            logging.info("Reading validated training data for transformation")
            train_df = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_file(self.data_validation_artifact.valid_test_file_path)

            x_train_df = train_df.drop(columns=[TARGET_COLUMN])
            y_train_series = train_df[TARGET_COLUMN].map({-1: 0.0, 1: 1.0}).astype(float)

            x_test_df = test_df.drop(columns=[TARGET_COLUMN])
            y_test_series = test_df[TARGET_COLUMN].map({-1: 0.0, 1: 1.0}).astype(float)

            return x_train_df, y_train_series, x_test_df, y_test_series
        except Exception as e:
            logging.error(f"Error occurred in get_train_test_df: {e}")
            raise NetException(e, sys) from e
        
    def pipeline_fit_transform(
            self,
            x_train_df: pd.DataFrame,
            x_test_df: pd.DataFrame,
            pipeline: Pipeline
    ) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
        try:
            logging.info("Fitting and transforming data...")
            preprocessor = pipeline.fit(x_train_df)
            transformed_x_train_df = preprocessor.transform(x_train_df)
            transformed_x_test_df = preprocessor.transform(x_test_df)

            return transformed_x_train_df, transformed_x_test_df, preprocessor
        except Exception as e:
            logging.error(f"Error occurred in pipeline_fit_transform: {e}")
            raise NetException(e, sys) from e
    
    def save_transformed_data(
            self,
            transformed_x_train_df: pd.DataFrame,
            transformed_x_test_df: pd.DataFrame,
            y_train_series: pd.Series,
            y_test_series: pd.Series
    ) -> tuple[str, str]:
        raise NotImplementedError("Subclasses should implement this method")
    
    def save_preprocessor_object(
            self,
            preprocessor: Pipeline
    ) -> str:
        try:
            logging.info("Saving preprocessor object...")
            object_file_path = self.data_transformation_config.preprocessor_object_file_path
            save_object(object_file_path, preprocessor)

            return object_file_path
        except Exception as e:
            logging.error(f"Error occurred in save_preprocessor_object: {e}")
            raise NetException(e, sys) from e
        
    def wrap_artifact(
            self,
            train_file_path: str,
            test_file_path: str,
            object_file_path: str
    ) -> Union[GenericDataTransformationArtifact, CatBoostDataTransformationArtifact]:
        raise NotImplementedError("Subclasses should implement this method")

    def initiate_data_transformation(
            self
    ) -> Union[GenericDataTransformationArtifact, CatBoostDataTransformationArtifact]:
        try:
            x_train_df, y_train_series, x_test_df, y_test_series = self.get_train_test_df()

            pipeline = self.pipeline_init()

            transformed_x_train_df, transformed_x_test_df, preprocessor = self.pipeline_fit_transform(
                x_train_df,
                x_test_df,
                pipeline
            )

            train_file_path, test_file_path = self.save_transformed_data(
                transformed_x_train_df,
                transformed_x_test_df,
                y_train_series,
                y_test_series
            )

            object_file_path = self.save_preprocessor_object(preprocessor)

            data_transformation_artifact = self.wrap_artifact(
                train_file_path,
                test_file_path,
                object_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_data_transformation: {e}")
            raise NetException(e, sys) from e
