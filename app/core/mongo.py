import os

from pymongo import MongoClient
from pymongo.database import Database


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "digital_twin")

_mongo_client = MongoClient(MONGO_URI)
mongo_db = _mongo_client[MONGO_DB_NAME]


def get_db() -> Database:
    return mongo_db


def initialize_indexes(db: Database) -> None:
    db.profiles.create_index("person_id", unique=True)
    db.profiles.create_index("last_seen_at")
    db.observations.create_index([("person_id", 1), ("captured_at", -1)])
    db.visits.create_index([("person_id", 1), ("entry_time", -1)])
