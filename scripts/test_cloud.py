#!/usr/bin/env python3
"""Test ChromaDB Cloud connection."""

from utils.cloud_storage import CloudChromaStorage


def test_cloud_connection():
    """Test connection to ChromaDB Cloud."""
    try:
        print("Testing ChromaDB Cloud connection...")
        storage = CloudChromaStorage(
            collection_name="test_collection"
        )
        
        # Test basic operations
        info = storage.get_collection_info()
        print(f"✅ Connected successfully!")
        print(f"Collection: {info['name']}")
        print(f"Count: {info['count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your cloud URL is correct")
        print("2. Make sure you're connected to internet")
        print("3. Verify your ChromaDB Cloud project is active")
        return False


if __name__ == "__main__":
    test_cloud_connection()
