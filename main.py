from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging

from src.pipeline.training_pipeline import TrainingPipeline

import sys

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        data_transformation_artifact = training_pipeline.data_processing_pipeline()
        model_trainer = training_pipeline.model_training_pipeline(data_transformation_artifact)
        print(model_trainer)
    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise NetException(e, sys)