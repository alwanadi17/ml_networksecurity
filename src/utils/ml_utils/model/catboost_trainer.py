from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.entity.artifact_entity import ModelTrainerArtifact
from src.utils.ml_utils.metric.classification_metric import classification_result

import os
import sys
import pandas as pd
from typing import Dict, Any
from catboost import CatBoostClassifier

class CatBoostModelTrainer:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact):
        self.model_trainer_artifact = model_trainer_artifact

    def train_catboost_model(self, 
                             x_train: pd.DataFrame, 
                             y_train: pd.Series, 
                             report_file_path: str
    ) -> None:
        try:
            logging.info("Training catboost model...")

            data_config: Dict = self.model_trainer_artifact.data_config
            cat_features = data_config['cat_features']
            params = {
                'cat_features': cat_features,
                'verbose': 0
            }
            model: CatBoostClassifier = self.model_trainer_artifact.model
            model.set_params(**params)

            model.fit(x_train, y_train)

            logging.info("Model trained successfully.")
            logging.info("Predicting on training data to evaluate performance...")

            y_pred_train = model.predict(x_train)
            train_classification_report_artifact = classification_result(y_train, y_pred_train, report_file_path)
            
            logging.info("Training results:")
            for metric, value in train_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            params = model.get_all_params()

            self.model_trainer_artifact = ModelTrainerArtifact(
                model=model,
                data_config=data_config,
                params=params,
                train_classification_report_artifact=train_classification_report_artifact
            )

        except Exception as e:
            logging.error(f"Error occurred in train_model: {e}")
            raise NetException(e, sys) from e
        
    def test_catboost_model(self,
                            x_test: pd.DataFrame,
                            y_test: pd.Series,
                            report_file_path: str
    ) -> None:
        try:
            logging.info("Testing model...")
            model: Any = self.model_trainer_artifact.model
            y_pred = model.predict(x_test)

            test_classification_report_artifact = classification_result(y_test, y_pred, report_file_path)
            logging.info("Testing results:")
            for metric, value in test_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            self.model_trainer_artifact.test_classification_report_artifact = test_classification_report_artifact
        except Exception as e:
            logging.error(f"Error occurred in test_model: {e}")
            raise NetException(e, sys) from e
        

    def initiate_catboost_trainer(self) -> ModelTrainerArtifact:
        try:
            pass
        except Exception as e:
            logging.error(f"Error occurred in initiate_catboost_trainer: {e}")
            raise NetException(e, sys)