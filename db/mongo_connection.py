# database.py
from mongoengine import connect, disconnect
import os
from dotenv import load_dotenv

load_dotenv()

def init_db():
    try:
        # Connect to MongoDB and specify the database name
        connect("canonicar", host=os.getenv("MONGO_URL"))
        return True
    except Exception as e:
        # If there's an error, print the exception and return None
        print(f"An error occurred while connecting to MongoDB: {e}")
        return None


# Close the connection to MongoDB
def close_db():
    try:
        disconnect("canonicar")
        print("Disconnected from MongoDB")
    except Exception as e:
        print(f"An error occurred while disconnecting from MongoDB: {e}")