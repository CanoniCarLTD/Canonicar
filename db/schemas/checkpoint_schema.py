from mongoengine import Document, StringField, DateTimeField, ReferenceField
from .episode_schema import Episode

class Checkpoint(Document):
    checkpoint_id = StringField(primary_key=True)
    episode = ReferenceField(Episode)  # Foreign key to Episode collection
    model_path = StringField(required=True)
    timestamp = DateTimeField(required=True)

    meta = {'collection': 'checkpoints'}
