from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.components.ingestion.data_ingestion import DataIngestion
from src.entity.config_entity import (
    TrainingPipelineEntity, 
    DataIngestionConfigEntity, 
    DataValidationConfigEntity, 
    CatBoostDataTransformationConfigEntity,
    GenericDataTransformationConfigEntity,
    # ModelTrainerConfigEntity
)
from src.components.validation.data_validation import DataValidation
from src.components.transformation.catboost_transformation import CatBoostDataTransformation
from src.components.transformation.generic_transformation import GenericDataTransformation
# from src.components.trainers.model_trainer import ModelTrainer

import sys

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipelineEntity()
        logging.info(f"Training pipeline initiated: {training_pipeline}")

        ingestion_config = DataIngestionConfigEntity(training_pipeline)
        logging.info(f"Data ingestion config initiated: {ingestion_config}")
        data_ingestion = DataIngestion(ingestion_config)
        logging.info(f"Data ingestion initiated: {data_ingestion}")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

        validation_config = DataValidationConfigEntity(training_pipeline)
        logging.info(f"Data validation config initiated: {validation_config}")
        data_validation = DataValidation(validation_config, data_ingestion_artifact)
        logging.info(f"Data validation initiated: {data_validation}")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data validation artifact: {data_validation_artifact}")

        catboost_transformation_config = CatBoostDataTransformationConfigEntity(training_pipeline)
        logging.info(f"Data transformation config initiated: {catboost_transformation_config}")
        catboost_data_transformation = CatBoostDataTransformation(catboost_transformation_config, data_validation_artifact)
        logging.info(f"Data transformation initiated: {catboost_data_transformation}")
        catboost_data_transformation_artifact = catboost_data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation artifact: {catboost_data_transformation_artifact}")

        generic_transformation_config = GenericDataTransformationConfigEntity(training_pipeline)
        logging.info(f"Data transformation config initiated: {generic_transformation_config}")
        generic_data_transformation = GenericDataTransformation(generic_transformation_config, data_validation_artifact)
        logging.info(f"Data transformation initiated: {generic_data_transformation}")
        generic_data_transformation_artifact = generic_data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation artifact: {generic_data_transformation_artifact}")

        # model_trainer_config = ModelTrainerConfigEntity(training_pipeline)
        # logging.info(f"Model trainer config initiated: {model_trainer_config}")
        # model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        # logging.info(f"Model trainer initiated: {model_trainer}")
        # model_trainer_artifact = model_trainer.initiate_model_trainer()
        # logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        
    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise NetException(e, sys)