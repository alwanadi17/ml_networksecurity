import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import ModelTrainerConfigEntity
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.utils.utils import read_parquet_file, read_yaml_file, write_yaml_file, save_object
from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.utils.ml_utils.model.catboost_trainer import train_catboost_model, test_catboost_model

class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfigEntity,
                 data_transformation_artifact: DataTransformationArtifact
                 ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact
        
    def test_model(self, model: CatBoostClassifier, x_test: pd.DataFrame, y_test: pd.Series, report_file_path: str) -> ClassificationReportArtifact:
        try:
            logging.info("Testing model...")
            y_pred = model.predict(x_test)

            test_classification_report_artifact = self.classification_result(y_test, y_pred, report_file_path)
            logging.info("Testing results:")
            for metric, value in test_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            return test_classification_report_artifact
        except Exception as e:
            logging.error(f"Error occurred in test_model: {e}")
            raise NetException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer...")
            transformed_x_train_df = read_parquet_file(self.data_transformation_artifact.transformed_train_file_path)
            x_train = transformed_x_train_df.drop(TARGET_COLUMN, axis=1)
            y_train = transformed_x_train_df[TARGET_COLUMN]
            
            transformed_x_test_df = read_parquet_file(self.data_transformation_artifact.transformed_test_file_path)
            x_test = transformed_x_test_df.drop(TARGET_COLUMN, axis=1)
            y_test = transformed_x_test_df[TARGET_COLUMN]

            data_config = read_yaml_file(self.data_transformation_artifact.data_config_file_path)
            logging.info("Data loaded successfully for model training.")

            logging.info("Training CatBoost model...")
            model_trainer_artifact = ModelTrainerArtifact(
                model=CatBoostClassifier(),
                data_config=data_config
            )
            train_report_file_path = self.model_trainer_config.classification_report_train_file_path
            test_report_file_path = self.model_trainer_config.classification_report_test_file_path
            model_artifact = train_catboost_model(x_train, y_train, train_report_file_path)
            test_classification_report_artifact = self.test_catboost_model(model, x_test, y_test, test_report_file_path)

            logging.info("Saving model...")
            save_object(self.model_trainer_config.model_file_path, model)
            write_yaml_file(self.model_trainer_config.model_config_file_path, model_config)

            model_trainer_artifact = ModelTrainerArtifact(
                model_file_path=self.model_trainer_config.model_file_path,
                model_config_file_path=self.model_trainer_config.model_config_file_path,
                classification_report_train_artifact=train_classification_report_artifact,
                classification_report_test_artifact=test_classification_report_artifact
            )

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_model_trainer: {e}")
            raise NetException(e, sys) from e