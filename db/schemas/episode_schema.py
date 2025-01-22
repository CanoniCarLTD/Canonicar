from mongoengine import Document, StringField, FloatField, IntField


class Episode(Document):
    episode_id = StringField(primary_key=True)
    reward = FloatField(required=True)
    average_reward = FloatField()
    policy_loss = FloatField()
    value_loss = FloatField()
    entropy = FloatField()
    duration = FloatField()
    num_steps = IntField()
    stage = StringField()

    meta = {"collection": "episodes"}
