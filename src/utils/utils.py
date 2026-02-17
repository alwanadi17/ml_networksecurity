from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging

import yaml
import dill
import os
import sys
import pandas as pd

def read_yaml_file(file_path: str)->dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logging.error(f"Error occurred while reading yaml file: {e}")
        raise NetException(e, sys)
    
def write_yaml_file(file_path: str, data: object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file)
        logging.info(f"Yaml file saved at: {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing yaml file: {e}")
        raise NetException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        logging.error(f"Error occured at save_object stage: {e}")
        raise NetException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        
        logging.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error occured at load_object stage: {e}")
        raise NetException(e, sys)
    
def read_csv_file(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"CSV file read successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error occurred while reading CSV file: {e}")
        raise NetException(e, sys)
    
def write_csv_file(file_path: str, df: pd.DataFrame):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False, header=True)
        logging.info(f"DataFrame written successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing DataFrame to CSV file: {e}")
        raise NetException(e, sys)
    
def write_parquet_file(file_path: str, df: pd.DataFrame):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path, index=False)
        logging.info(f"DataFrame written successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while writing DataFrame to Parquet file: {e}")
        raise NetException(e, sys)