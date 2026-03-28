from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

def get_filtered_data(limit=None):
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client["your_database_name"]  # Replace with your database name
        collection = db["rides"]

        # Define query (example query for last 7 days)
        time_threshold = datetime.now() - timedelta(days=7)
        query = {"datestamp": {"$gte": int(time_threshold.timestamp())}}

        # Retrieve data with optional limit
        if limit:
            data = list(collection.find(query).limit(limit))
        else:
            data = list(collection.find(query))

        print(f"Retrieved {len(data)} records from the database.")
        return data
    except Exception as e:
        print(f"Error fetching data from MongoDB: {str(e)}")
        return []
