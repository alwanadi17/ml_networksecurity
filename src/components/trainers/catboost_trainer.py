from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.components.trainers.model_trainer import ModelTrainer
from src.entity.artifact_entity import ModelTrainerArtifact, CatBoostDataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfigEntity
from src.utils.utils import read_parquet_file, read_yaml_file
from src.constant.training_pipeline import TARGET_COLUMN

import os
import sys
import pandas as pd
from typing import Any, Dict
from catboost import CatBoostClassifier

class CatBoostModelTrainer(ModelTrainer):
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfigEntity,
            catboost_data_transformation_artifact: CatBoostDataTransformationArtifact
    ):
        super().__init__(
            model_trainer_config=model_trainer_config
        )
        self.catboost_data_transformation_artifact = catboost_data_transformation_artifact

    def initiate_catboost_trainer(
            self,
            model_obj: CatBoostClassifier,
            params: Dict[str, Any]
    ) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading catboost data...")
            train_df = read_parquet_file(self.catboost_data_transformation_artifact.transformed_train_file_path)
            test_df = read_parquet_file(self.catboost_data_transformation_artifact.transformed_test_file_path)
            data_config = read_yaml_file(self.catboost_data_transformation_artifact.data_config_file_path)
            logging.info(f"Loading catboost data completed.")

            x_train_df = train_df.drop(columns=[TARGET_COLUMN])
            y_train_df = train_df[TARGET_COLUMN]

            x_test_df = test_df.drop(columns=[TARGET_COLUMN])
            y_test_df = test_df[TARGET_COLUMN]

            params = {
                'cat_features': data_config['cat_features'],
                'verbose': 0
            }

            model_trainer_artifact = self.initiate_model_trainer(
                x_train=x_train_df,
                y_train=y_train_df,
                x_test=x_test_df,
                y_test=y_test_df,
                 model_obj=model_obj,
                params=params
            )

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_catboost_trainer: {e}")
            raise NetException(e, sys)