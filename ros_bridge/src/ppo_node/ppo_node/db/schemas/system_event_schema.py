from mongoengine import Document, StringField, DateTimeField


class SystemEvent(Document):
    event_id = StringField(primary_key=True)
    timestamp = DateTimeField(required=True)
    component = StringField(required=True)
    message = StringField(required=True)

    meta = {"collection": "system_events"}
