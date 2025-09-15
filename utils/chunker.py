from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# Initialize tokenizer for chunking
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
MAX_TOKENS = 1000
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

tokenizer.model_max_length = MAX_TOKENS

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

# Initialize embedding model for generating embeddings
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def chunk_document(docling_document: DoclingDocument):
    """Chunk a document and return chunks with contextualized text."""
    
    chunk_iter = chunker.chunk(dl_doc=docling_document)
    chunks = []
    
    for chunk in chunk_iter:
        # Use contextualize() method as recommended in docling docs
        contextualized_text = chunker.contextualize(chunk=chunk)
        # Create a new chunk object with contextualized text
        chunk.text = contextualized_text
        chunks.append(chunk)
    
    return chunks


