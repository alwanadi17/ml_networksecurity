# common constants for training pipeline

TARGET_COLUMN: str = "Result"
TRAINING_PIPELINE_NAME: str = "net_sec_training_pipeline"
ARTIFACT_DIR_NAME: str = "artifacts"
FILE_DIR: str = "data/raw"
FILE_NAME: str = "phisingData.csv"

RAW_FILE_NAME: str = "raw.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Data Ingestion Contants
DATA_INGESTION_DATABASE_NAME: str = "ALWAN_NETSEC_AI"
DATA_INGESTION_COLLECTION_NAME: str = "NetworkSecurityData"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# Data Validation Constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"