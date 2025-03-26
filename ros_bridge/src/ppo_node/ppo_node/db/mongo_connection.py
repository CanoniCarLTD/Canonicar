# database.py
from mongoengine import connect, disconnect
from pymongo import MongoClient
import os
from dotenv import load_dotenv



def init_db():
    load_dotenv()
    try:
        # Use Atlas connection string from env
        client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        db = client.get_database(os.getenv("MONGO_DB"))
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
