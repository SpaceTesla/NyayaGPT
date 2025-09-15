"""Configuration settings for NyayaGPT."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    max_tokens: int = int(os.getenv('MAX_TOKENS', 1000))
    overlap_size: int = 50
    merge_peers: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class StorageConfig:
    """Configuration for vector storage."""
    collection_name: str = os.getenv('COLLECTION_NAME', 'documents')
    persist_directory: str = os.getenv('CHROMA_DB_DIR', './chroma_db')
    distance_metric: str = "cosine"
    batch_size: int = int(os.getenv('BATCH_SIZE', '250'))
    # Cloud configuration
    api_key: str = os.getenv('CHROMA_API_KEY', '')
    tenant: str = os.getenv('CHROMA_TENANT', '')
    database: str = os.getenv('CHROMA_DATABASE', '')


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    file: str = os.getenv('LOG_FILE', 'logs/nyayagpt.log')
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


@dataclass
class AppConfig:
    """Main application configuration."""
    name: str = os.getenv('APP_NAME', 'NyayaGPT')
    version: str = os.getenv('APP_VERSION', '0.1.0')
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Sub-configurations
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    storage: StorageConfig = None
    logging: LoggingConfig = None
    
    # Document processing
    supported_formats: list = None
    data_dir: str = os.getenv('DATA_DIR', 'data')
    
    def __post_init__(self):
        # Initialize sub-configurations
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.storage is None:
            self.storage = StorageConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".docx", ".txt", ".md"]


# Global configuration instance
config = AppConfig()
