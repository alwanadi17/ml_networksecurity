from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import TrainingPipelineEntity, DataIngestionConfigEntity

import sys

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipelineEntity()
        ingestion_config = DataIngestionConfigEntity(training_pipeline)
        data_ingestion = DataIngestion(ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise NetException(e, sys)