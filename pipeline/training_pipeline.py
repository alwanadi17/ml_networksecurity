from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException as NetException
from src.entity.config_entity import (
    DataIngestionConfigEntity,
    DataValidationConfigEntity,
    DataTransformationConfigEntity,
    ModelTrainerConfigEntity,
    TrainingPipelineEntity
)

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from src.components.ingestion.data_ingestion import DataIngestion
from src.components.validation.data_validation import DataValidation
from src.components.transformation.data_transformation import DataTransformation
# from src.components.trainers.base_model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(
            self
    ):
        pass

    def _generate_config_entity(
            self
    ):
        try:
            training_pipeline = TrainingPipelineEntity()
            logging.info(f"Training pipeline initiated: {training_pipeline}")

            self.data_ingestion_config = DataIngestionConfigEntity(training_pipeline)
            self.data_validation_config = DataValidationConfigEntity(training_pipeline)
            self.data_transformation_config = DataTransformationConfigEntity(training_pipeline)
            # self.model_trainer_config = ModelTrainerConfigEntity(training_pipeline)
        except Exception as e:
            logging.error(f"Error in _generate_config_entity: {e}")
            raise NetException(e)

    def data_processing_pipeline(
            self
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data processing pipeline.")
            self._generate_config_entity()
            # Data Ingestion
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            # Data Validation
            data_validation = DataValidation(self.data_validation_config, data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()

            # Data Transformation
            data_transformation = DataTransformation(self.data_transformation_config, data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data processing pipeline completed successfully.")
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")

            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error in data_processing_pipeline: {e}")
            raise NetException(e)