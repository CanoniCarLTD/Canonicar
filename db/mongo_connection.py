# database.py
from pymongo import MongoClient, ASCENDING
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global MongoDB client
_client = None
_db = None

def get_client():
    """Get singleton MongoDB client"""
    global _client
    if _client is None:
        init_db()
    return _client

def get_db():
    """Get MongoDB database"""
    global _db
    if _db is None:
        init_db()
    return _db

def init_db():
    """Initialize MongoDB connection"""
    global _client, _db
    
    load_dotenv()
    try:
        print("hello from init_db")
        # Use Atlas connection string from env
        mongo_url = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
        _client = MongoClient(
            mongo_url,
            retryWrites=True,
            w="majority",
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        _client.admin.command('ping')
        
        # Set database
        _db = _client["canonicar"]
        
        # Initialize collections with indexes
        _db.episodes.create_index([("episode_id", ASCENDING)], unique=True)
        _db.training_metrics.create_index([("episode", ASCENDING), ("step", ASCENDING)])
        _db.checkpoints.create_index([("checkpoint_id", ASCENDING)], unique=True)
        _db.performance.create_index([("performance_id", ASCENDING)], unique=True)
        _db.error_logs.create_index([("timestamp", ASCENDING)])
        
        logger.info("Successfully connected to MongoDB Atlas")
        return True
    except Exception as e:
        logger.error(f"An error occurred while connecting to MongoDB Atlas: {e}")
        _client = None
        _db = None
        return False

def close_db():
    """Close MongoDB connection"""
    global _client, _db
    try:
        if _client:
            _client.close()
            _client = None
            _db = None
            logger.info("Disconnected from MongoDB")
    except Exception as e:
        logger.error(f"An error occurred while disconnecting from MongoDB: {e}")
