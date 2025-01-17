from mongoengine import Document, StringField, DateTimeField, BooleanField

class ErrorLog(Document):
    error_id = StringField(primary_key=True)
    timestamp = DateTimeField(required=True)
    component = StringField(required=True)
    message = StringField(required=True)
    resolution_status = BooleanField()

    meta = {'collection': 'error_logs'}
