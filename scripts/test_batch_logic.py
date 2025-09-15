#!/usr/bin/env python3
"""Test the batch upload logic to ensure it handles all chunks correctly."""

def test_batch_logic():
    """Test the batch upload logic with different scenarios."""
    
    # Test with 378 chunks (Indian Constitution)
    total_chunks = 378
    batch_size = 250
    
    print(f"Testing with {total_chunks} chunks and batch_size={batch_size}")
    print("=" * 50)
    
    uploaded = 0
    batch_count = 0
    
    for i in range(0, total_chunks, batch_size):
        batch = list(range(i, min(i + batch_size, total_chunks)))
        batch_count += 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"Batch {batch_count}/{total_batches}: Chunks {i+1}-{min(i+batch_size, total_chunks)}")
        print(f"  Batch size: {len(batch)} chunks")
        print(f"  Chunk IDs: {batch[0]+1} to {batch[-1]+1}")
        
        uploaded += len(batch)
        print(f"  Total uploaded so far: {uploaded}/{total_chunks}")
        print()
    
    print(f"🎉 Final result: {uploaded}/{total_chunks} chunks would be uploaded")
    print(f"✅ All chunks covered: {uploaded == total_chunks}")
    
    # Test edge cases
    print("\n" + "=" * 50)
    print("Testing edge cases:")
    
    test_cases = [
        (9, 250),    # Resume case
        (300, 250),  # Exactly at limit
        (301, 250),  # Just over limit
        (500, 250),  # Well over limit
    ]
    
    for chunks, batch_size in test_cases:
        batches_needed = (chunks + batch_size - 1) // batch_size
        print(f"  {chunks} chunks with batch_size={batch_size} → {batches_needed} batches")


if __name__ == "__main__":
    test_batch_logic()
