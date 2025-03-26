from pymongo import MongoClient, ASCENDING

# PyMongo schema definition (for reference)
EPISODE_SCHEMA = {
    "episode_id": str,  # primary key
    "reward": float,    # required
    "average_reward": float,
    "policy_loss": float,
    "value_loss": float,
    "entropy": float,
    "duration": float,
    "num_steps": int,
    "stage": str
}

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['your_database_name']
episodes_collection = db['episodes']

# Create index for episode_id
def setup_indexes():
    episodes_collection.create_index([("episode_id", ASCENDING)], unique=True)

# Example function to insert an episode
def insert_episode(episode_data):
    episodes_collection.insert_one(episode_data)

# Example function to find an episode by episode_id
def find_episode(episode_id):
    return episodes_collection.find_one({"episode_id": episode_id})

# Example usage
if __name__ == "__main__":
    setup_indexes()
    episode_data = {
        "episode_id": "ep1",
        "reward": 10.0,
        "average_reward": 8.0,
        "policy_loss": 0.5,
        "value_loss": 0.3,
        "entropy": 0.1,
        "duration": 120.0,
        "num_steps": 100,
        "stage": "training"
    }
    insert_episode(episode_data)
    print(find_episode("ep1"))
