#!/usr/bin/env python3
"""Migration script to move from local ChromaDB to cloud."""

import chromadb
from chromadb.config import Settings
from utils import ChromaStorage
import os


def migrate_to_cloud(cloud_url: str, api_key: str = None):
    """Migrate local data to ChromaDB Cloud."""
    
    # Connect to local database
    print("Connecting to local database...")
    local_storage = ChromaStorage(collection_name="resume_chunks")
    
    # Get all data
    print("Exporting local data...")
    all_data = local_storage.get_all_data()
    
    print(f"Found {len(all_data['ids'])} chunks to migrate")
    
    # Connect to cloud
    print("Connecting to ChromaDB Cloud...")
    cloud_client = chromadb.HttpClient(
        host=cloud_url,
        port=8000,
        settings=Settings(allow_reset=True)
    )
    
    # Create cloud collection
    cloud_collection = cloud_client.get_or_create_collection(
        name="resume_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Migrate data
    print("Migrating data to cloud...")
    cloud_collection.add(
        ids=all_data['ids'],
        documents=all_data['documents'],
        embeddings=all_data['embeddings'],
        metadatas=all_data['metadatas']
    )
    
    print(f"✅ Successfully migrated {len(all_data['ids'])} chunks to cloud!")
    print(f"Cloud collection: {cloud_collection.name}")
    print(f"Cloud count: {cloud_collection.count()}")


def test_cloud_connection(cloud_url: str):
    """Test connection to ChromaDB Cloud."""
    try:
        client = chromadb.HttpClient(host=cloud_url, port=8000)
        collections = client.list_collections()
        print(f"✅ Connected to cloud! Found {len(collections)} collections")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Replace with your actual cloud URL
    CLOUD_URL = "your-chroma-cloud-url.chromadb.com"
    
    print("Testing cloud connection...")
    if test_cloud_connection(CLOUD_URL):
        print("\nStarting migration...")
        migrate_to_cloud(CLOUD_URL)
    else:
        print("Please check your cloud URL and try again.")
