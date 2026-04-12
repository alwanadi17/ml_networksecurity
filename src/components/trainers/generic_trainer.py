from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.components.trainers.base_model_trainer import BaseModelTrainer
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfigEntity
from src.utils.utils import read_npy_file

import os
import sys
from typing import Any, Dict

class GenericModelTrainer(BaseModelTrainer):
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfigEntity,
            data_transformation_artifact: DataTransformationArtifact,
            model_class: Any,
            params: Dict[str, Any]
    ):
        super().__init__(
            model_trainer_config=model_trainer_config,
            model_class=model_class,
            params=params
        )
        self.generic_data_transformation_artifact = data_transformation_artifact.generic_data_transformation_artifact

    def initiate_model_trainer(
            self
    ) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading data for generic model...")
            train_npy = read_npy_file(self.generic_data_transformation_artifact.transformed_train_file_path)
            test_npy = read_npy_file(self.generic_data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Loading data completed.")

            x_train = train_npy[:, :-1]
            y_train = train_npy[:, -1]
            x_test = test_npy[:, :-1]
            y_test = test_npy[:, -1]

            model_trainer_artifact = self.training_model_pipe(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_model_trainer: {e}")
            raise NetException(e, sys)