from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv, find_dotenv
import os
import sys
import certifi

from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging

ca = certifi.where()

load_dotenv(find_dotenv())

uri = os.environ.get('MONGO_DB_URI')

# Create a new client and connect to the server
client = MongoClient(
    uri,
    tlsCAFile=ca,
    server_api=ServerApi('1')
)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logging.error(f"Error occurred while connecting to MongoDB: {e}")
    raise NetException(e, sys)