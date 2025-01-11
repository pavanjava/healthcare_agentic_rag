import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed.text import TextEmbedding
from dotenv import load_dotenv, find_dotenv
from typing import List
import openlit

openlit.init(otlp_endpoint="http://127.0.0.1:4318")


# print(TextEmbedding.list_supported_models())


class MedicalData:

    def __init__(self):
        load_dotenv(find_dotenv())
        self.qdrant_client = QdrantClient(url=os.environ.get('QDRANT_URL'), api_key=os.environ.get('QDRANT_API_KEY'))
        self.embedding_model = TextEmbedding(model_name='snowflake/snowflake-arctic-embed-m')
        self.records: List[str] = []

    def _load_data(self):
        """Load medical data from a JSON file and store in records list."""
        with open("data/medical_data.json", "r") as f:
            data = json.load(f)

        for category in data["data"]:
            for key, case_list in category.items():
                self.records.extend(case["case"] for case in case_list)

        print(f"Loaded {len(self.records)} medical case records.")

    def check_and_create_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection('medical_records')
            print("LOG: Collection 'zoom_recordings' already exists")
        except Exception as e:
            print("LOG: Creating collection 'zoom_recordings'...")
            self.qdrant_client.create_collection(
                collection_name='medical_records',
                vectors_config=models.VectorParams(
                    size=768,  # depends on the model's output vector dimension
                    distance=models.Distance.COSINE
                )
            )
            print("LOG: Collection created successfully")

    def insert_to_collection(self):
        """Insert records into the Qdrant collection."""
        embeddings = self.embedding_model.embed(self.records)
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload={"text": text},
            )
            for idx, (embedding, text) in enumerate(zip(embeddings, self.records))
        ]

        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.qdrant_client.upsert(
                    collection_name='medical_records',
                    points=batch
                )
                print(f"LOG: Inserted batch {i // batch_size + 1} of {len(points) // batch_size + 1}")
            except Exception as e:
                print(f"LOG: Error inserting batch: {e}")

        print("LOG: Collection population complete")
        print(f"Inserted {len(self.records)} records into the collection.")

    def search_collection(self, query: str, limit: int = 10):
        """ search for medical history in vector database"""
        print(f"LOG: Searching medical history with query: {query}")
        query_embedding = self.embedding_model.embed(query).tolist()
        # Search Qdrant with limit of 10
        print("LOG: Searching Qdrant")
        vector_results = self.qdrant_client.search(
            collection_name='zoom_recordings',
            query_vector=query_embedding,
            limit=limit,  # Changed from default to 10
            score_threshold=0.7  # Only return good matches
        )

        return vector_results


if __name__ == "__main__":
    data_ingester = MedicalData()
    data_ingester._load_data()
    data_ingester.check_and_create_collection()
    data_ingester.insert_to_collection()
