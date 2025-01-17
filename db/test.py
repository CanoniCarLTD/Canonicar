from db.mongo_connection import init_db
from db.schemas import Episode, Checkpoint, Performance, ErrorLog, SystemEvent
from datetime import datetime


# Call the init_db function to connect to MongoDB
if init_db():
    # Create the first Episode object
    episode = Episode(
        episode_id="episode_1",
        reward=100.0,
        average_reward=90.0,
        policy_loss=0.2,
        value_loss=0.1,
        entropy=0.05,
        duration=120.5,
        num_steps=1000,
        stage="Training"
    )

    # Save the object to the MongoDB database
    episode.save()
    
    timestamp_str = "2025-01-17T00:00:00Z"
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

    checkpoint = Checkpoint(
    checkpoint_id="checkpoint_001",
    episode=episode,  # Linking the checkpoint to the episode
    model_path="/path/to/model",
    timestamp=timestamp
    )
    checkpoint.save()

    performance = Performance(
    performance_id="performance_001",
    race_id="race_001",
    agent_type="reinforcement_learning",
    lap_times=[15.2, 14.8, 14.5],
    total_time=45.0,
    path_deviation=0.5,
    collisions=2,
    lap_progress=0.9
    )   
    performance.save()

    error_log = ErrorLog(
        error_id="error_001",
        timestamp=timestamp, 
        component="training_module",
        message="Model overfitting detected.",
        resolution_status=False  # False means not resolved yet
    )
    error_log.save()

    system_event = SystemEvent(
    event_id="event_001",
    timestamp=timestamp,
    component="carla_simulator",
    message="Simulation started successfully."
    )
    system_event.save()



else:
    print("Failed to connect to the database.")
