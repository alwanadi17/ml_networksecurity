# common constants for training pipeline
import os
import numpy as np

TARGET_COLUMN: str = "Result"
TRAINING_PIPELINE_NAME: str = "net_sec_training_pipeline"
ARTIFACT_DIR_NAME: str = "artifacts"
FILE_DIR: str = "data/raw"
FILE_NAME: str = "phisingData.csv"

RAW_FILE_NAME: str = "raw.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_DIR_NAME: str = "schema"
SCHEMA_FILE_NAME: str = "schema.yaml"

SCHEMA_FILE_PATH: str = os.path.join(SCHEMA_DIR_NAME, SCHEMA_FILE_NAME)

# Data Ingestion Contants
DATA_INGESTION_DATABASE_NAME: str = "ALWAN_NETSEC_AI"
DATA_INGESTION_COLLECTION_NAME: str = "NetworkSecurityData"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# Data Validation Constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR_NAME: str = "valid"
DATA_VALIDATION_INVALID_DIR_NAME: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR_NAME: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"

# Data Transformation Constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_OBJECT_DIR_NAME: str = "objects"
DATA_TRANSFORMATION_GENERIC_OBJECT_FILE_NAME: str = "generic_preprocessor.pkl"
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME: str = "data_transformed"
DATA_TRANSFORMATION_TRANSFORMED_GENERIC_DIR_NAME: str = "data_transformed_generic"
DATA_TRANSFORMATION_TRANSFORMED_GENERIC_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TRANSFORMED_GENERIC_TEST_FILE_NAME: str = "test.npy"

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "n_neighbors": 3,
    "weights": "uniform",
    "missing_values": np.nan
}

DATA_TRANSFORMATION_CATBOOST_DATA_CONFIG_DIR: str = "data_config"
DATA_TRANSFORMATION_CATBOOST_DATA_CONFIG_FILE_NAME: str = "data_config.yaml"
DATA_TRANSFORMATION_CATBOOST_OBJECT_FILE_NAME: str = "catboost_preprocessor.pkl"
DATA_TRANSFORMATION_TRANSFORMED_CATBOOST_DIR_NAME: str = "data_transformed_catboost"
DATA_TRANSFORMATION_TRANSFORMED_CATBOOST_TRAIN_FILE_NAME: str = "train.parquet"
DATA_TRANSFORMATION_TRANSFORMED_CATBOOST_TEST_FILE_NAME: str = "test.parquet"

# Model Trainer Constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_MODEL_TRAINED_DIR_NAME: str = "models"
MODEL_TRAINER_MODEL_PARAMS_DIR_NAME: str = "model_params"
MODEL_TRAINER_MODEL_PARAMS_FILE_NAME: str = "model_params.yaml"
MODEL_TRAINER_CLASSIFICATION_REPORT_DIR_NAME: str = "classification_report"
MODEL_TRAINER_CLASSIFICATION_REPORT_TRAIN_FILE_NAME: str = "classification_report_train.yaml"
MODEL_TRAINER_CLASSIFICATION_REPORT_TEST_FILE_NAME: str = "classification_report_test.yaml"