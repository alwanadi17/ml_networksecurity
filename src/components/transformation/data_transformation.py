import os
import sys
from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from src.utils.utils import read_yaml_file
from src.entity.config_entity import (
    DataTransformationConfigEntity,
    GenericDataTransformationConfigEntity,
    CatBoostDataTransformationConfigEntity
)
from src.components.transformation.generic_transformation import GenericDataTransformation
from src.components.transformation.catboost_transformation import CatBoostDataTransformation

class DataTransformation:
    def __init__(
            self,
            data_transformation_config: DataTransformationConfigEntity,
            data_validation_artifact: DataValidationArtifact,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact

    def initiate_data_transformation(
            self
    ) -> DataTransformationArtifact:
        try:
            models_config = read_yaml_file(self.data_transformation_config.models_config_file_path)
            tp_entity = self.data_transformation_config.tp_entity
            generic_data_transformation_artifact = None
            catboost_data_transformation_artifact = None
            is_generated_generic = False

            # Only run transformations for the models specified in the config file
            for model_type in models_config:
                if model_type == 'CatBoost':
                    catboost_transformation_config = CatBoostDataTransformationConfigEntity(tp_entity)
                    catboost_transformation = CatBoostDataTransformation(catboost_transformation_config, self.data_validation_artifact)
                    catboost_data_transformation_artifact = catboost_transformation.initiate_data_transformation()
                elif is_generated_generic == False and model_type != 'CatBoost':
                    generic_transformation_config = GenericDataTransformationConfigEntity(tp_entity)
                    generic_transformation = GenericDataTransformation(generic_transformation_config, self.data_validation_artifact)
                    generic_data_transformation_artifact = generic_transformation.initiate_data_transformation()
                    is_generated_generic = True

            data_transformation_artifact = DataTransformationArtifact(
                generic_data_transformation_artifact=generic_data_transformation_artifact,
                catboost_data_transformation_artifact=catboost_data_transformation_artifact
            )

            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_data_transformation: {e}")
            raise NetException(e, sys) from e