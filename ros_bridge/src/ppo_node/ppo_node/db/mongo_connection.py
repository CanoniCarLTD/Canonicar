# database.py
from mongoengine import connect, disconnect
from pymongo import MongoClient
import os
from dotenv import load_dotenv

def init_db():
    load_dotenv(dotenv_path="/ros_bridge/src/.env")
    try:
        conn_str = os.getenv("MONGO_CONNECTION_STRING")
        print(f"print the conn str: {conn_str}")
        client = MongoClient(conn_str)
        db_name = os.getenv("MONGO_DB")
        print(f"db name: {conn_str}")        
        db = client[db_name]
        print("✅ Successfully connected to MongoDB Atlas")
        return db
    except Exception as e:
        print(f"An error occurred while connecting to MongoDB Atlas: {e}")
        return None


# Close the connection to MongoDB
def close_db():
    try:
        disconnect("canonicar")
        print("Disconnected from MongoDB")
    except Exception as e:
        print(f"An error occurred while disconnecting from MongoDB: {e}")
