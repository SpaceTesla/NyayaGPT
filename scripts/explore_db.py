#!/usr/bin/env python3
"""Database explorer for ChromaDB collections."""

from utils import ChromaStorage
import json


def main():
    """Explore the ChromaDB database."""
    # Connect to existing database
    storage = ChromaStorage(collection_name="resume_chunks")
    
    # Show collection info
    info = storage.get_collection_info()
    print(f"Collection: {info['name']}")
    print(f"Total chunks: {info['count']}")
    print(f"Metadata: {info['metadata']}")
    
    # Explore all chunks
    print("\n" + "="*60)
    storage.explore_database(limit=20)  # Show more chunks
    
    # Get raw data for inspection
    print("\n" + "="*60)
    print("Raw data structure:")
    all_data = storage.get_all_data()
    print(f"Keys: {list(all_data.keys())}")
    print(f"Number of IDs: {len(all_data['ids'])}")
    print(f"Number of documents: {len(all_data['documents'])}")
    print(f"Number of metadatas: {len(all_data['metadatas'])}")
    
    # Show first chunk's full metadata
    if all_data['metadatas']:
        print(f"\nFirst chunk metadata: {json.dumps(all_data['metadatas'][0], indent=2)}")


if __name__ == "__main__":
    main()
