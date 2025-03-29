# database.py
from mongoengine import connect, disconnect
from pymongo import MongoClient
import os

def init_db():
    try:
        conn_str = os.getenv("MONGO_CONNECTION_STRING")
        client = MongoClient(conn_str)
        db_name = os.getenv("MONGO_DB")
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
