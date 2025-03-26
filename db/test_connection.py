import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

def init_db():
    try:
        load_dotenv()
        client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        db = client.get_database(os.getenv("MONGO_DB"))
        print("✅ Successfully connected to MongoDB Atlas")
        return db
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB Atlas: {e}")
        return None

def close_db(client):
    client.close()

def test_atlas_connection():
    db = init_db()
    if db is not None:
        try:
            # Try to create a test document
            test_episode = {
                "episode_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "reward": 100.0,
                "num_steps": 1000,
                "stage": "test"
            }
            result = db.episodes.insert_one(test_episode)
            
            print("✅ Successfully saved test episode to MongoDB Atlas")
            
            # Clean up test data
            db.episodes.delete_one({"_id": result.inserted_id})
            close_db(db.client)
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_atlas_connection()