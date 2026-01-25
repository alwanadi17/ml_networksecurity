import sys
import os
import json
import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv
import certifi
import pymongo
from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging

load_dotenv(find_dotenv())

MONGO_DB_URI = os.environ.get('MONGO_DB_URI')

ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            logging.error(f"Error occurred in NetworkDataExtract initialization: {e}")
            raise NetException(e, sys)
        
    def csv_to_json(self, file_path: str) -> str:
        """
        Converts a CSV file to JSON format.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            str: The JSON string representation of the CSV data.
        """
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            data = df.to_json(orient='records')
            return data
        except Exception as e:
            logging.error(f"Error occurred while converting CSV to JSON: {e}")
            raise NetException(e, sys)
        

    def push_data_to_mongodb(self, data: list, db_name: str, collection_name: str):
        """
        Pushes data to a MongoDB collection.

        Args:
            data (list): The data to be inserted into the collection.
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.

        Returns:
            int: The number of documents inserted.
        """
        try:
            self.db_name = db_name
            self.collection_name = collection_name
            self.data = data

            self.client = pymongo.MongoClient(MONGO_DB_URI, tlsCAFile=ca)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.collection.insert_many(self.data)

            logging.info(f"Data pushed to MongoDB collection: {collection_name} in database: {db_name} length: {len(data)}")
            return len(self.data)
        except Exception as e:
            logging.error(f"Error occurred while pushing data to MongoDB: {e}")
            raise NetException(e, sys)
        

if __name__ == "__main__":
    try:
        FILE_PATH = "data/raw/phisingData.csv"
        DB = "ALWAN_NETSEC_AI"
        COLLECTION = "NetworkSecurityData"
        extractor = NetworkDataExtract()
        json_data = extractor.csv_to_json(file_path=FILE_PATH)
        data_list = json.loads(json_data)
        inserted_count = extractor.push_data_to_mongodb(data=data_list, db_name=DB, collection_name=COLLECTION)
        print(f"Number of documents inserted: {inserted_count}")
    except Exception as e:
        logging.error(f"Error occurred in test push main function: {e}")
        raise NetException(e, sys)