#!/usr/bin/env python3
"""Clear the ChromaDB collection to start fresh."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cloud_storage import CloudChromaStorage
from config.config import config


def clear_collection():
    """Clear the current collection."""
    print("Connecting to ChromaDB Cloud...")
    storage = CloudChromaStorage(
        collection_name=config.storage.collection_name
    )
    
    # Get current collection info
    info = storage.get_collection_info()
    print(f"Current collection: {info['name']}")
    print(f"Current count: {info['count']} chunks")
    
    if info['count'] > 0:
        print("\nClearing collection...")
        try:
            # Delete the collection
            storage.client.delete_collection(storage.collection.name)
            print("✅ Collection cleared successfully!")
            
            # Recreate empty collection
            storage.collection = storage.client.create_collection(
                name=config.storage.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ New empty collection created!")
            
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
    else:
        print("Collection is already empty!")


if __name__ == "__main__":
    clear_collection()
