from flask_pymongo import PyMongo

mongo = PyMongo()


def init_db(app):
    print("Connecting to MongoDB...")
    mongo.init_app(app, uri=app.config["MONGO_URI"])
