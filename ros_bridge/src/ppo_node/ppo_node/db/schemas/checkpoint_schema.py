from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from ppo_node.mongo_connection import init_db, close_db

# Initialize MongoDB connection
init_db()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost:27017/"))
db = client['canonicar']  # Replace with your database name if different
checkpoints_collection = db['checkpoints']

class Checkpoint:
    def __init__(self, checkpoint_id, episode_id, model_path, timestamp):
        self.checkpoint_id = checkpoint_id
        self.episode_id = episode_id
        self.model_path = model_path
        self.timestamp = timestamp

    def save(self):
        checkpoint_data = {
            "_id": self.checkpoint_id,
            "episode_id": ObjectId(self.episode_id),
            "model_path": self.model_path,
            "timestamp": self.timestamp
        }
        checkpoints_collection.insert_one(checkpoint_data)

    @staticmethod
    def get_by_id(checkpoint_id):
        data = checkpoints_collection.find_one({"_id": checkpoint_id})
        if data:
            return Checkpoint(
                checkpoint_id=data["_id"],
                episode_id=data["episode_id"],
                model_path=data["model_path"],
                timestamp=data["timestamp"]
            )
        return None

# Example usage
checkpoint = Checkpoint(
    checkpoint_id="some_id",
    episode_id="some_episode_id",
    model_path="/path/to/model",
    timestamp=datetime.now()
)
checkpoint.save()

# Close the MongoDB connection
close_db()
