from datetime import datetime
import os
import src.constant.training_pipeline as tp

class TrainingPipelineEntity:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = tp.TRAINING_PIPELINE_NAME
        self.artifact_dir_name = tp.ARTIFACT_DIR_NAME
        self.artifact_dir = os.path.join(self.artifact_dir_name, self.pipeline_name, self.timestamp)


class DataIngestionConfigEntity:
    def __init__(self, tp_entity: TrainingPipelineEntity):
        self.database_name = tp.DATA_INGESTION_DATABASE_NAME
        self.collection_name = tp.DATA_INGESTION_COLLECTION_NAME
        self.data_ingestion_dir_name = tp.DATA_INGESTION_DIR_NAME
        self.feature_store_dir_name = tp.DATA_INGESTION_FEATURE_STORE_DIR_NAME
        self.ingested_dir_name = tp.DATA_INGESTION_INGESTED_DIR_NAME
        self.train_test_split_ratio = tp.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

        self.data_file_path = os.path.join(
            tp.FILE_DIR,
            tp.FILE_NAME
        )

        self.data_ingestion_artifact_dir = os.path.join(
            tp_entity.artifact_dir,
            self.data_ingestion_dir_name
        )

        self.feature_store_file_path = os.path.join(
            self.data_ingestion_artifact_dir,
            self.feature_store_dir_name,
            tp.RAW_FILE_NAME
        )

        self.train_file_path = os.path.join(
            self.data_ingestion_artifact_dir,
            self.ingested_dir_name,
            tp.TRAIN_FILE_NAME
        )

        self.test_file_path = os.path.join(
            self.data_ingestion_artifact_dir,
            self.ingested_dir_name,
            tp.TEST_FILE_NAME
        )

class DataValidationConfigEntity:
    def __init__(self, tp_entity: TrainingPipelineEntity):
        self.data_validation_dir_name = tp.DATA_VALIDATION_DIR_NAME
        self.valid_dir_name = tp.DATA_VALIDATION_VALID_DIR_NAME
        self.invalid_dir_name = tp.DATA_VALIDATION_INVALID_DIR_NAME
        self.drift_report_dir_name = tp.DATA_VALIDATION_DRIFT_REPORT_DIR_NAME

        self.data_validation_artifact_dir = os.path.join(
            tp_entity.artifact_dir,
            self.data_validation_dir_name
        )

        self.valid_train_file_path = os.path.join(
            self.data_validation_artifact_dir,
            self.valid_dir_name,
            tp.TRAIN_FILE_NAME
        )

        self.valid_test_file_path = os.path.join(
            self.data_validation_artifact_dir,
            self.valid_dir_name,
            tp.TEST_FILE_NAME
        )

        self.invalid_train_file_path = os.path.join(
            self.data_validation_artifact_dir,
            self.invalid_dir_name,
            tp.TRAIN_FILE_NAME
        )

        self.invalid_test_file_path = os.path.join(
            self.data_validation_artifact_dir,
            self.invalid_dir_name,
            tp.TEST_FILE_NAME
        )

        self.drift_report_file_path = os.path.join(
            self.data_validation_artifact_dir,
            self.drift_report_dir_name,
            tp.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


