#!/usr/bin/env python3
"""Test Pinecone connection and search functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pinecone_storage import PineconeStorage
from utils.chunker import embedding_model
import time

def test_pinecone():
    """Test Pinecone functionality."""
    print("🔍 Testing Pinecone Connection and Search")
    print("=" * 50)
    
    try:
        # Initialize storage
        storage = PineconeStorage(
            index_name="nyayagpt-constitution",
            dimension=1024
        )
        
        # Get index info
        print("\n📊 Index Information:")
        info = storage.get_index_info()
        print(f"Name: {info.get('name', 'N/A')}")
        print(f"Dimension: {info.get('dimension', 'N/A')}")
        print(f"Total vectors: {info.get('total_vector_count', 'N/A')}")
        print(f"Namespaces: {info.get('namespaces', 'N/A')}")
        
        # Wait a bit for indexing
        print("\n⏳ Waiting 5 seconds for indexing to complete...")
        time.sleep(5)
        
        # Test search without filter
        print("\n🔍 Testing search without filter:")
        results = storage.search("fundamental rights", n_results=3)
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. ID: {result['id']}")
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Text: {result['text'][:100]}...")
            print(f"   Document: {result['metadata'].get('document_name', 'N/A')}")
        
        # Test search with filter
        print("\n🔍 Testing search with document filter:")
        results = storage.search("fundamental rights", n_results=3, document_name="indian_constitution")
        print(f"Found {len(results)} results with filter")
        
        # Test a simple query
        print("\n🔍 Testing simple query:")
        results = storage.search("constitution", n_results=2)
        print(f"Found {len(results)} results for 'constitution'")
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. ID: {result['id']}")
            print(f"   Text: {result['text'][:100]}...")
        
        # Test embedding generation
        print("\n🧠 Testing embedding generation:")
        test_embedding = embedding_model.encode(["test query"])
        print(f"Embedding shape: {test_embedding.shape}")
        print(f"Embedding type: {type(test_embedding)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pinecone()
