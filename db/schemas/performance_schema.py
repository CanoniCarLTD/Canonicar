from mongoengine import Document, StringField, FloatField, IntField, ListField

class Performance(Document):
    performance_id = StringField(primary_key=True)
    race_id = StringField(required=True)
    agent_type = StringField(required=True)
    lap_times = ListField(FloatField())
    total_time = FloatField()
    path_deviation = FloatField()
    collisions = IntField()
    lap_progress = FloatField()

    meta = {'collection': 'performance'}
