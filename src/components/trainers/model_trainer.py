from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.entity.artifact_entity import ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfigEntity
from src.utils.ml_utils.metrics.classification_metric import classification_result
from src.utils.utils import save_object, write_yaml_file

import os
import sys
import pandas as pd
from typing import Dict, Any

class ModelTrainer:
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfigEntity
    ):
        self.model_trainer_config = model_trainer_config

    def _generate_model_paths(self, model: Any) -> Dict[str, str]:
        try:
            model_name = model.__class__.__name__
            model_trained_dir = os.path.join(self.model_trainer_config.model_trained_dir, model_name)
            classification_report_dir = os.path.join(model_trained_dir, self.model_trainer_config.classification_report_dir_name)

            model_file_path = os.path.join(model_trained_dir, f"{model_name}.pkl")
            model_params_file_path = os.path.join(model_trained_dir, self.model_trainer_config.model_params_file_name)
            train_report_file_path = os.path.join(classification_report_dir, self.model_trainer_config.classification_report_train_file_name)
            test_report_file_path = os.path.join(classification_report_dir, self.model_trainer_config.classification_report_test_file_name)

            return {
                model_file_path: model_file_path,
                model_params_file_path: model_params_file_path,
                train_report_file_path: train_report_file_path,
                test_report_file_path: test_report_file_path
            }
        except Exception as e:
            logging.error(f"Error occurred in _generate_model_paths: {e}")
            raise NetException(e, sys)

    def train_model(
            x_train: pd.DataFrame,
            y_train: pd.Series,
            model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer_artifact.model.set_params(**model_trainer_artifact.params)
            model_trainer_artifact.model.fit(x_train, y_train)

            logging.info("Predicting on training data to evaluate performance...")

            y_pred_train = model_trainer_artifact.model.predict(x_train)
            train_classification_report_artifact, report_dict = classification_result(y_train, y_pred_train)
            
            logging.info("Training results:")
            for metric, value in train_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            logging.info("Saving model training results...")
            write_yaml_file(report_dict, model_trainer_artifact.train_report_file_path)
            logging.info(f"Report saved at: {model_trainer_artifact.train_report_file_path}")

            model_trainer_artifact.train_classification_report_artifact = train_classification_report_artifact

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in train_model: {e}")
            raise NetException(e, sys) from e
        
    def test_model(
            x_test: pd.DataFrame,
            y_test: pd.Series,
            model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Testing model...")
            y_pred = model_trainer_artifact.model.predict(x_test)

            test_classification_report_artifact, report_dict = classification_result(y_test, y_pred)
            logging.info("Testing results:")
            for metric, value in test_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            logging.info("Saving model testing results...")
            write_yaml_file(report_dict, model_trainer_artifact.test_report_file_path)
            logging.info(f"Report saved at: {model_trainer_artifact.test_report_file_path}")

            model_trainer_artifact.test_classification_report_artifact = test_classification_report_artifact

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in test_model: {e}")
            raise NetException(e, sys) from e

    def initiate_model_trainer(
            self, 
            x_train: Any, 
            y_train: Any,
            x_test: Any,
            y_test: Any,
            model_obj: Any,
            params: Dict[str, Any]
    ) -> ModelTrainerArtifact:
        try:
            logging.info(f"--- Process started for model: {model_obj.__class__.__name__}")

            paths = self._generate_model_paths(model_obj)

            model_trainer_artifact = ModelTrainerArtifact(
                model=model_obj,
                params=params,
                **paths
            )

            model_trainer_artifact = self.train_model(x_train, y_train, model_trainer_artifact)
            model_trainer_artifact = self.test_model(x_test, y_test, model_trainer_artifact)
            logging.info("Model trained successfully.")

            logging.info("Saving model...")
            save_object(model_trainer_artifact.model_file_path, model_trainer_artifact.model)
            write_yaml_file(model_trainer_artifact.model_params_file_path, model_trainer_artifact.params)

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_model_trainer: {e}")
            raise NetException(e, sys) from e
