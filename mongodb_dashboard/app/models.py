from pymongo import MongoClient
import os
import time
from datetime import datetime, timedelta
import logging

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

def get_filtered_data(limit=None):
    # Ensure environment variable is loaded and MongoDB URI is set
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable is not set.")

    client = MongoClient(mongo_uri)
    db = client['your_database_name']  # Replace with your database name
    collection = db['rides']  # Replace with your collection name

    # Example query: Filter records within the last 7 days
    time_threshold = int((datetime.now() - timedelta(days=7)).timestamp())
    query = {"datestamp": {"$gte": time_threshold}}

    # Fetch data with or without a limit
    if limit:
        data = list(collection.find(query).limit(limit))
    else:
        data = list(collection.find(query))

    # Log the number of retrieved records
    logging.debug(f"Retrieved {len(data)} records")

    return data
