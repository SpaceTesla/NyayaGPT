"""Data validation utilities for NyayaGPT."""

from typing import List, Dict, Any
import numpy as np
from .embedder import ChunkWithEmbedding


def validate_chunks(chunks: List[ChunkWithEmbedding]) -> List[str]:
    """Validate chunks and return list of issues."""
    issues = []
    
    if not chunks:
        issues.append("No chunks provided")
        return issues
    
    for i, chunk in enumerate(chunks):
        # Check text
        if not chunk.text or not chunk.text.strip():
            issues.append(f"Chunk {i}: Empty text")
        
        # Check embedding
        if not hasattr(chunk, 'embedding') or chunk.embedding is None:
            issues.append(f"Chunk {i}: Missing embedding")
        elif not isinstance(chunk.embedding, np.ndarray):
            issues.append(f"Chunk {i}: Embedding is not numpy array")
        elif chunk.embedding.size == 0:
            issues.append(f"Chunk {i}: Empty embedding")
        elif len(chunk.embedding.shape) != 1:
            issues.append(f"Chunk {i}: Embedding should be 1D array")
    
    return issues


def validate_search_results(results: List[Dict[str, Any]]) -> List[str]:
    """Validate search results and return list of issues."""
    issues = []
    
    if not results:
        issues.append("No search results")
        return issues
    
    for i, result in enumerate(results):
        required_fields = ['id', 'text', 'distance', 'metadata']
        for field in required_fields:
            if field not in result:
                issues.append(f"Result {i}: Missing field '{field}'")
        
        if 'distance' in result and not isinstance(result['distance'], (int, float)):
            issues.append(f"Result {i}: Distance should be numeric")
    
    return issues


def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Validate metadata for ChromaDB compatibility."""
    issues = []
    
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            issues.append(f"Metadata key '{key}' has invalid type: {type(value)}")
    
    return issues
