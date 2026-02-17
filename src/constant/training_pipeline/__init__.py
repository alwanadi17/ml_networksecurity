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
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME: str = "data_transformed"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME: str = "train.parquet"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME: str = "test.parquet"
DATA_TRANSFORMATION_OBJECT_DIR_NAME: str = "objects"
DATA_TRANSFORMATION_OBJECT_FILE_NAME: str = "preprocessor.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "n_neighbors": 3,
    "weights": "uniform",
    "missing_values": np.nan
}