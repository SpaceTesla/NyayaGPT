from utils.chunker import chunk_document, embedding_model
from docling_core.types.doc import DoclingDocument
import numpy as np
from dataclasses import dataclass
from typing import List, Any


@dataclass
class ChunkWithEmbedding:
    """Container for chunk data with embedding."""
    text: str
    embedding: np.ndarray
    metadata: dict = None


def embed_document(docling_document):
    """Chunk a document and generate embeddings for each chunk."""
    # docling_document = DoclingDocument.load_from_json(docling_document)
    chunks = chunk_document(docling_document)
    
    # Generate embeddings for all chunks
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.encode(texts)
    
    # Create chunks with embeddings
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedded_chunk = ChunkWithEmbedding(
            text=chunk.text,
            embedding=embeddings[i],
            metadata={
                'chunk_id': i,
                'chunk_type': type(chunk).__name__
            }
        )
        embedded_chunks.append(embedded_chunk)
    
    return embedded_chunks


    