from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.components.trainers.model_trainer import ModelTrainer
from src.entity.artifact_entity import ModelTrainerArtifact, GenericDataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfigEntity
from src.utils.utils import read_npy_file

import os
import sys
from typing import Any, Dict

class GenericModelTrainer(ModelTrainer):
    def __init__(
            self,
            model_trainer_config: ModelTrainerConfigEntity,
            generic_data_transformation_artifact: GenericDataTransformationArtifact
    ):
        super().__init__(
            model_trainer_config=model_trainer_config
        )
        self.generic_data_transformation_artifact = generic_data_transformation_artifact

    def initiate_generic_trainer(
            self,
            model_obj: Any,
            params: Dict[str, Any]
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

            model_trainer_artifact = self.initiate_model_trainer(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                model_obj=model_obj,
                params=params
            )

            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_generic_trainer: {e}")
            raise NetException(e, sys)