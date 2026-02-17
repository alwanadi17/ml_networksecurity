from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import (
    TrainingPipelineEntity, 
    DataIngestionConfigEntity, 
    DataValidationConfigEntity, 
    DataTransformationConfigEntity
)
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

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

        transformation_config = DataTransformationConfigEntity(training_pipeline)
        logging.info(f"Data transformation config initiated: {transformation_config}")
        data_transformation = DataTransformation(transformation_config, data_validation_artifact)
        logging.info(f"Data transformation initiated: {data_transformation}")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation artifact: {data_transformation_artifact}")
        
    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise NetException(e, sys)