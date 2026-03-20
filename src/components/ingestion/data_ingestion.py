from src.exception.exception import NetworkSecurityException as NetSecException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfigEntity
from src.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv, find_dotenv
import pymongo
import certifi

ca = certifi.where()

load_dotenv(find_dotenv())

MONGO_DB_URI = os.getenv("MONGO_DB_URI")

class DataIngestion:
    def __init__(self, ingestion_config:DataIngestionConfigEntity):
        try:
            self.ingestion_config = ingestion_config
        except Exception as e:
            logging.error(f"Error occurred in DataIngestion initialization: {e}")
            raise NetSecException(e, sys) from e
        
    def export_collection_to_dataframe(self):
        try:
            # client = pymongo.MongoClient(MONGO_DB_URI, tlsCAFile=ca)
            # db = client[self.ingestion_config.database_name]
            # collection = db[self.ingestion_config.collection_name]
            # df = pd.DataFrame(list(collection.find()))
            # logging.info(f"df imported from collection: {self.ingestion_config.collection_name} in database: {self.ingestion_config.database_name}")

            # id = '_id'
            # if id in df.columns:
            #     df = df.drop(columns=[id], axis=1)

            df = pd.read_csv(self.ingestion_config.data_file_path)

            logging.info(f"Returning df... Dataframe shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error occurred while exporting collection to dataframe: {e}")
            raise NetSecException(e, sys) from e
        
    def export_data_to_feature_store(self):
        try:
            df = self.export_collection_to_dataframe()

            feature_store_dir = os.path.dirname(self.ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            df.to_csv(self.ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info(f"Data raw exported at {self.ingestion_config.feature_store_file_path}")

            return self.ingestion_config.feature_store_file_path
        except Exception as e:
            logging.error(f"Error occurred while exporting data to feature store: {e}")
            raise NetSecException(e, sys) from e
        
    def split_data_as_train_test(self):
        try:
            df = pd.read_csv(self.ingestion_config.feature_store_file_path)

            logging.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(
                df,
                test_size=self.ingestion_config.train_test_split_ratio,
                random_state=42
            )
            logging.info(f"Data split completed. Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            logging.info("Saving train and test sets to respective file paths...")
            os.makedirs(os.path.dirname(self.ingestion_config.train_file_path), exist_ok=True)

            train_set.to_csv(self.ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_file_path, index=False, header=True)

            logging.info(f"Data split into train and test sets at:")
            logging.info(f"train_set_path: {self.ingestion_config.train_file_path}")
            logging.info(f"test_set_path: {self.ingestion_config.test_file_path}")
            return (
                self.ingestion_config.train_file_path,
                self.ingestion_config.test_file_path
            )
        except Exception as e:
            logging.error(f"Error occurred while splitting data into train and test sets: {e}")
            raise NetSecException(e, sys) from e
        
    def initiate_data_ingestion(self):
        try:
            raw_file_path = self.export_data_to_feature_store()
            train_file_path, test_file_path = self.split_data_as_train_test()

            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=raw_file_path,
                train_file_path=train_file_path,
                test_file_path=test_file_path
            )

            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Error occurred in data ingestion: {e}")
            raise NetSecException(e, sys) from e