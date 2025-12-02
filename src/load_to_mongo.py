
import pandas as pd
from pymongo import MongoClient
import os

def load_data_to_mongodb(csv_path, db_name, collection_name, mongo_uri="mongodb://localhost:27017/"):
    """
    Loads data from a CSV file into a MongoDB collection.

    Args:
        csv_path (str): The path to the CSV file.
        db_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.
        mongo_uri (str): The MongoDB connection URI.
    """
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Read the CSV file
        if not os.path.exists(csv_path):
            print(f"Error: The file was not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)

        # Convert dataframe to a list of dictionaries (records)
        data = df.to_dict(orient='records')

        # Insert data into the collection
        if data:
            collection.delete_many({})  # Clear existing data
            collection.insert_many(data)
            print(f"Successfully loaded {len(data)} records into '{db_name}.{collection_name}'")
        else:
            print("No data to load.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths and names
    CSV_FILE_PATH = os.path.join(project_root, 'data', 'hateXplain_cleaned.csv')
    DB_NAME = "CyberBullying"
    COLLECTION_NAME = "messages"

    # Load the data
    load_data_to_mongodb(CSV_FILE_PATH, DB_NAME, COLLECTION_NAME)
