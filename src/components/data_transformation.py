import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

from sklearn import set_config
set_config(transform_output='pandas')

from src.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.config_entity import DataTransformationConfigEntity
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.utils.utils import save_object, read_csv_file, write_parquet_file, write_yaml_file

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfigEntity,
                 data_validation_artifact: DataValidationArtifact):
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

    def pipeline_init(self) -> Pipeline:
        try:
            logging.info("Preparing data transformation pipeline")
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            pipeline = Pipeline(steps=[
                ('imputer', imputer)
            ])

            return pipeline
        except Exception as e:
            logging.error(f"Error occurred in initiate_pipeline: {e}")
            raise NetException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
          try:
            train_df = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_file(self.data_validation_artifact.valid_test_file_path)

            x_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_train_df = train_df[TARGET_COLUMN]

            x_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test_df = test_df[TARGET_COLUMN]

            pipeline = self.pipeline_init()

            logging.info("Fitting and transforming data...")
            preprocessor = pipeline.fit(x_train_df)
            transformed_x_train_df = pipeline.transform(x_train_df)
            transformed_x_test_df = pipeline.transform(x_test_df)            

            logging.info("Saving data config...")
            categorical_cols = pd.DataFrame(transformed_x_train_df).columns
            data_config = {
                'categorical_cols': list(categorical_cols)
            }
            data_config_file_path = self.data_transformation_config.data_config_file_path
            write_yaml_file(data_config_file_path, data_config)

            logging.info("Saving transformed data...")
            train_concatenated_df = pd.concat([transformed_x_train_df, y_train_df], axis=1)
            test_concatenated_df = pd.concat([transformed_x_test_df, y_test_df], axis=1)

            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_config.transformed_test_file_path

            write_parquet_file(transformed_train_file_path, train_concatenated_df)
            write_parquet_file(transformed_test_file_path, test_concatenated_df)

            logging.info("Saving preprocessor object...")
            object_file_path = self.data_transformation_config.preprocessor_object_file_path
            save_object(object_file_path, preprocessor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                transformed_object_file_path=object_file_path,
                data_config_file_path=data_config_file_path
            )

            return data_transformation_artifact

          except Exception as e:
            logging.error(f"Error occurred in initiate_data_transformation: {e}")
            raise NetException(e, sys) from e