from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging

from src.pipeline.training_pipeline import TrainingPipeline

import sys

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        data_transformation_artifact = training_pipeline.data_processing_pipeline()
        top_k_models = training_pipeline.model_training_pipeline(data_transformation_artifact)
        logging.info(f"\n\ntop_k_models:{top_k_models}")
    except Exception as e:
        logging.error(f"Error occurred in main function: {e}")
        raise NetException(e, sys)