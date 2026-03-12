import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, precision_score, recall_score

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import ModelTrainerConfigEntity
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationReportArtifact
from src.utils.utils import read_parquet_file, read_yaml_file, write_yaml_file, save_object
from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException

class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfigEntity,
                 data_transformation_artifact: DataTransformationArtifact
                 ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def classification_result(self, y_true: pd.Series, y_pred: pd.Series, report_file_path:str) -> ClassificationReportArtifact:
        try:
            accuracy = accuracy_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            classification_report_dict = classification_report(y_true, y_pred, output_dict=True)

            report_to_save = {
                'overall_accuracy': accuracy,
                'overall_roc_auc': roc_auc,
                'overall_f1_score': f1,
                'overall_precision': precision,
                'overall_recall': recall,
                'macro_avg': classification_report_dict['macro avg'],
                'weighted_avg': classification_report_dict['weighted avg'],
                'class_specific': {
                    'attack': classification_report_dict['-1'],
                    'normal': classification_report_dict['1']
                }
            }

            logging.info(f"Saving classification report at {report_file_path}")
            write_yaml_file(report_file_path, report_to_save)

            classification_report_artifact = ClassificationReportArtifact(
                report_file_path=report_file_path,
                accuracy=accuracy,
                roc_auc=roc_auc,
                f1_score=f1,
                precision=precision,
                recall=recall
            )

            return classification_report_artifact
        except Exception as e:
            logging.error(f"Error occurred in classification_result: {e}")
            raise NetException(e, sys) from e

    def train_model(self, x_train: pd.DataFrame, y_train: pd.Series, data_config: dict, report_file_path: str) -> Tuple[CatBoostClassifier, dict, ClassificationReportArtifact]:
        try:
            logging.info("Training model...")
            model = CatBoostClassifier(cat_features=data_config['categorical_cols'], verbose=0)
            model.fit(x_train, y_train)
            logging.info("Model trained successfully.")
            logging.info("Predicting on training data to evaluate performance...")

            y_pred_train = model.predict(x_train)
            train_classification_report_artifact = self.classification_result(y_train, y_pred_train, report_file_path)
            
            logging.info("Training results:")
            for metric, value in train_classification_report_artifact.__dict__.items():
                if metric != 'report_file_path':
                    logging.info(f"  {metric.capitalize()}: {value}")

            return model, model.get_all_params(), train_classification_report_artifact
        except Exception as e:
            logging.error(f"Error occurred in train_model: {e}")
            raise NetException(e, sys) from e
        
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

            train_report_file_path = self.model_trainer_config.classification_report_train_file_path
            test_report_file_path = self.model_trainer_config.classification_report_test_file_path
            model, model_config, train_classification_report_artifact = self.train_model(x_train, y_train, data_config, train_report_file_path)
            test_classification_report_artifact = self.test_model(model, x_test, y_test, test_report_file_path)

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