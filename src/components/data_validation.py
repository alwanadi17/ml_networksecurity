from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfigEntity
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.utils import read_yaml_file, write_yaml_file

import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self, validation_config:DataValidationConfigEntity, ingestion_artifact:DataIngestionArtifact):
        try:
            self.validation_config = validation_config
            self.ingestion_artifact = ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            logging.error(f"Error occurred in DataValidation initialization: {e}")
            raise NetException(e, sys) from e
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error occurred in read_data staticmethod: {e}")
            raise NetException(e, sys)
        
    def validate_len_columns(self, df:pd.DataFrame)->bool:
        try:
            schema_len_columns = len(self._schema_config)
            df_len_columns = len(df.columns)
            logging.info(f"Number of columns in schema file: {schema_len_columns}")
            logging.info(f"Number of columns in dataframe: {df_len_columns}")

            return schema_len_columns == df_len_columns
        except Exception as e:
            logging.error(f"Error occurred in validate_columns_number method: {e}")
            raise NetException(e, sys)
        
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05)->bool:
        try:
            status = True
            report = {}

            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]

                pval = ks_2samp(d1, d2).pvalue

                if pval > threshold:
                    is_drift = False
                else:
                    is_drift = True
                    status = False

                report.update({col: {"p_value": pval, "drift_status": is_drift}})

            drift_report_file_path = self.validation_config.drift_report_file_path

            write_yaml_file(drift_report_file_path, report)

            return status
        except Exception as e:
            logging.error(f"Error occurred in detect_dataset_drift method: {e}")
            raise NetException(e, sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.ingestion_artifact.train_file_path
            test_file_path = self.ingestion_artifact.test_file_path

            df_train = self.read_data(train_file_path)
            df_test = self.read_data(test_file_path)

            ## Schema Validation
            col_status = True
            validate_train_columns = self.validate_len_columns(df_train)
            if not validate_train_columns:
                logging.info("Validation Error: df_train columns do not match")
            
            validate_test_columns = self.validate_len_columns(df_test)
            if not validate_test_columns:
                logging.info("Validation Error: df_test columns do not match")

            validate_len_columns = validate_train_columns and validate_test_columns
            logging.info(f"Columns validation status: {validate_len_columns}")

            if not validate_len_columns:
                col_status = False

            ## Drift Validation
            drift_status = self.detect_dataset_drift(df_train, df_test)
            logging.info(f"Drift status: {drift_status}")

            status = col_status and drift_status
            logging.info(f"Data validation status: {status}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.validation_config.valid_train_file_path,
                valid_test_file_path=self.validation_config.valid_test_file_path,
                invalid_train_file_path=self.validation_config.invalid_train_file_path,
                invalid_test_file_path=self.validation_config.invalid_test_file_path,
                drift_report_file_path=self.validation_config.drift_report_file_path
            )

            if status:
                os.makedirs(self.validation_config.valid_dir, exist_ok=True)
                
                df_train.to_csv(self.validation_config.valid_train_file_path, index=False, header=True)
                df_test.to_csv(self.validation_config.valid_test_file_path, index=False, header=True)

                data_validation_artifact.invalid_test_file_path = None
                data_validation_artifact.invalid_train_file_path = None
            else:
                os.makedirs(self.validation_config.invalid_dir, exist_ok=True)

                df_train.to_csv(self.validation_config.invalid_train_file_path, index=False, header=True)
                df_test.to_csv(self.validation_config.invalid_test_file_path, index=False, header=True)

                data_validation_artifact.valid_test_file_path = None
                data_validation_artifact.valid_train_file_path = None

            return data_validation_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_data_validation: {e}")
            raise NetException(e, sys)