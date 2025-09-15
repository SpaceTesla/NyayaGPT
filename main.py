from utils.pinecone_storage import PineconeStorage


def main():
    # Initialize Pinecone storage (data already uploaded)
    print("--- Connecting to Pinecone ---")
    storage = PineconeStorage(
        index_name="nyayagpt-constitution",
        dimension=1024
    )
    
    # Show index info
    info = storage.get_index_info()
    print(f"Index info: {info}")
    
    # Test search
    print("\n--- Testing Search ---")
    search_results = storage.search("fundamental rights", n_results=3)
    print(f"Found {len(search_results)} results for 'fundamental rights':")
    for i, result in enumerate(search_results):
        print(f"\n{i+1}. Distance: {result['distance']:.4f}")
        print(f"   Text: {result['text'][:150]}...")
    
    # Test another query
    print("\n--- Testing Another Query ---")
    search_results = storage.search("constitution", n_results=2)
    print(f"Found {len(search_results)} results for 'constitution':")
    for i, result in enumerate(search_results):
        print(f"\n{i+1}. Distance: {result['distance']:.4f}")
        print(f"   Text: {result['text'][:150]}...")
    
    # Explore database contents
    storage.explore_database(limit=5)


if __name__ == "__main__":
    main()
