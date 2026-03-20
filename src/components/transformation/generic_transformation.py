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
from src.utils.utils import save_object, read_csv_file, write_npy_file
from src.components.processing.custom_transformer import ColumnDatatypeTransformer
        
class GenericDataTransformation:
    def __init__(self, 
                 data_transformation_config: GenericDataTransformationConfigEntity,
                 data_validation_artifact: DataValidationArtifact,
    ):
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

    def pipeline_init(self) -> Pipeline:
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

    def initiate_data_transformation(self) -> GenericDataTransformationArtifact:
          try:
            train_df = read_csv_file(self.data_validation_artifact.valid_train_file_path)
            test_df = read_csv_file(self.data_validation_artifact.valid_test_file_path)

            x_train_df = train_df.drop(columns=[TARGET_COLUMN])
            y_train_df = train_df[TARGET_COLUMN]

            x_test_df = test_df.drop(columns=[TARGET_COLUMN])
            y_test_df = test_df[TARGET_COLUMN]

            y_train_arr = y_train_df.values
            y_test_arr = y_test_df.values

            pipeline = self.pipeline_init()

            logging.info("Fitting and transforming data...")
            preprocessor = pipeline.fit(x_train_df)
            transformed_x_train_arr = preprocessor.transform(x_train_df)
            transformed_x_test_arr = preprocessor.transform(x_test_df)

            logging.info("Saving transformed data...")
            train_concatenated_arr = np.c_[transformed_x_train_arr, y_train_arr]
            test_concatenated_arr = np.c_[transformed_x_test_arr, y_test_arr]

            generic_transformed_train_file_path = self.data_transformation_config.generic_transformed_train_file_path
            generic_transformed_test_file_path = self.data_transformation_config.generic_transformed_test_file_path

            write_npy_file(generic_transformed_train_file_path, train_concatenated_arr)
            write_npy_file(generic_transformed_test_file_path, test_concatenated_arr)

            logging.info("Saving preprocessor object...")
            object_file_path = self.data_transformation_config.generic_preprocessor_object_file_path
            save_object(object_file_path, preprocessor)

            data_transformation_artifact = GenericDataTransformationArtifact(
                transformed_train_file_path=generic_transformed_train_file_path,
                transformed_test_file_path=generic_transformed_test_file_path,
                transformed_object_file_path=object_file_path
            )

            return data_transformation_artifact

          except Exception as e:
            logging.error(f"Error occurred in initiate_data_transformation: {e}")
            raise NetException(e, sys) from e