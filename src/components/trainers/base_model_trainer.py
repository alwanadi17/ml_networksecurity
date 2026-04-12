from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.entity.artifact_entity import ModelTrainerArtifact, ClassificationReportArtifact
from src.entity.config_entity import ModelTrainerConfigEntity
from src.utils.ml_utils.metrics.classification_metric import classification_result
from src.utils.utils import save_object, write_yaml_file, read_yaml_file

import os
import sys
import pandas as pd
from typing import Dict, Any

class BaseModelTrainer:
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfigEntity,
            model_class: Any,
            params: Dict[str, Any]
    ):
        self.model_trainer_config = model_trainer_config
        self._models_config = read_yaml_file(self.model_trainer_config.models_config_file_path)

        self.model = model_class()
        self.params = params

    def _generate_model_paths(
            self,
            model: Any
    ) -> Dict[str, str]:
        try:
            self.model_trainer_config.model_name = model.__class__.__name__
            self.model_trainer_config.model_trained_dir = os.path.join(
                self.model_trainer_config.model_trained_dir,
                self.model_trainer_config.model_name
            )
            self.model_trainer_config.classification_report_dir = os.path.join(
                self.model_trainer_config.model_trained_dir,
                self.model_trainer_config.classification_report_dir_name
            )
            self.model_trainer_config.model_file_path = os.path.join(
                self.model_trainer_config.model_trained_dir,
                f"{self.model_trainer_config.model_name}.pkl"
            )
            self.model_trainer_config.model_params_file_path = os.path.join(
                self.model_trainer_config.model_trained_dir,
                self.model_trainer_config.model_params_file_name
            )
            self.model_trainer_config.train_report_file_path = os.path.join(
                self.model_trainer_config.classification_report_dir,
                self.model_trainer_config.classification_report_train_file_name
            )
            self.model_trainer_config.test_report_file_path = os.path.join(
                self.model_trainer_config.classification_report_dir,
                self.model_trainer_config.classification_report_test_file_name
            )
        except Exception as e:
            logging.error(f"Error occurred in _generate_model_paths at BaseModelTrainer class: {e}")
            raise NetException(e, sys)

    def train_model(
            self,
            x_train: pd.DataFrame,
            y_train: pd.Series
    ) -> ClassificationReportArtifact:
        try:
            logging.info("Training model...")
            self.model.set_params(**self.params)
            self.model.fit(x_train, y_train)

            logging.info("Predicting on training data to evaluate performance...")

            y_pred_train = self.model.predict(x_train)
            train_classification_report_artifact, report_dict = classification_result(
                y_true=y_train, 
                y_pred=y_pred_train
            )
            
            logging.info("Training results:")
            for metric, value in train_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            logging.info("Saving model training results...")
            write_yaml_file(
                self.model_trainer_config.train_report_file_path,
                report_dict
            )
            logging.info(f"Report saved at: {self.model_trainer_config.train_report_file_path}")

            logging.info("Saving model...")
            save_object(self.model_trainer_config.model_file_path, self.model)

            logging.info("Update and save training params...")
            self.params = self.model.get_params()
            write_yaml_file(
                self.model_trainer_config.model_params_file_path,
                self.params
            )

            return train_classification_report_artifact
        except Exception as e:
            logging.error(f"Error occurred in train_model at BaseModelTrainer class: {e}")
            raise NetException(e, sys) from e
        
    def test_model(
            self,
            x_test: pd.DataFrame,
            y_test: pd.Series
    ) -> ClassificationReportArtifact:
        try:
            logging.info("Testing model...")
            y_pred = self.model.predict(x_test)

            test_classification_report_artifact, report_dict = classification_result(y_test, y_pred)
            logging.info("Testing results:")
            for metric, value in test_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            logging.info("Saving model testing results...")
            write_yaml_file(
                self.model_trainer_config.test_report_file_path,
                report_dict
            )
            logging.info(f"Report saved at: {self.model_trainer_config.test_report_file_path}")

            return test_classification_report_artifact
        except Exception as e:
            logging.error(f"Error occurred in test_model at BaseModelTrainer class: {e}")
            raise NetException(e, sys) from e

    def training_model_pipe(
            self,
            x_train: Any,
            y_train: Any,
            x_test: Any,
            y_test: Any
    ) -> ModelTrainerArtifact:
        try:
            logging.info(f"--- Process started for model: {self.model}")

            self._generate_model_paths(self.model)

            train_classification_report_artifact = self.train_model(x_train, y_train)
            test_classification_report_artifact = self.test_model(x_test, y_test)
            logging.info("Model trained successfully.")

            model_trainer_artifact = ModelTrainerArtifact(
                model_name=self.model_trainer_config.model_name,
                params=self.params,
                train_classification_report_artifact=train_classification_report_artifact,
                test_classification_report_artifact=test_classification_report_artifact,
                model_file_path=self.model_trainer_config.model_file_path,
                model_params_file_path=self.model_trainer_config.model_params_file_path,
                train_report_file_path=self.model_trainer_config.train_report_file_path,
                test_report_file_path=self.model_trainer_config.test_report_file_path
            )

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_model_trainer at BaseModelTrainer class: {e}")
            raise NetException(e, sys) from e
        
    def initiate_model_trainer(
            self
    ) -> ModelTrainerArtifact:
        raise NotImplementedError("Subclasses must implement this method.")
