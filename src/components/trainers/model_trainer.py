import os
import sys
from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException

from src.components.trainers.catboost_trainer import CatBoostModelTrainer
from src.components.trainers.generic_trainer import GenericModelTrainer
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfigEntity
from src.utils.utils import read_yaml_file
from src.utils.ml_utils.utils import import_class
from typing import Dict

class ModelTrainer:
    def __init__(
            self,
            data_transformation_artifact: DataTransformationArtifact,
            model_trainer_config: ModelTrainerConfigEntity
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(
            self
    ) -> Dict[str, ModelTrainerArtifact]:
        try:
            models = read_yaml_file(self.model_trainer_config.models_config_file_path)
            model_trainer_artifacts = {}

            for model_name, model_config in models.items():
                model_class = import_class(model_config['class'])
                params = model_config['params']
                if model_name == 'CatBoost':
                    catboost_model_trainer = CatBoostModelTrainer(
                        model_trainer_config=self.model_trainer_config,
                        data_transformation_artifact=self.data_transformation_artifact,
                        model_class=model_class,
                        params=params
                    )
                    model_trainer_artifact = catboost_model_trainer.initiate_model_trainer()
                else:
                    generic_model_trainer = GenericModelTrainer(
                        model_trainer_config=self.model_trainer_config,
                        data_transformation_artifact=self.data_transformation_artifact,
                        model_class=model_class,
                        params=params
                    )
                    model_trainer_artifact = generic_model_trainer.initiate_model_trainer()

                model_trainer_artifacts[model_name] = model_trainer_artifact

            return model_trainer_artifacts
        except Exception as e:
            logging.error(f"Error occurred in initiate_model_trainer at ModelTrainer class: {e}")
            raise NetException(e, sys) from e