"""
Configuration file for RAG pipeline performance optimizations.
"""

# Embedding Configuration
EMBEDDING_CONFIG = {
    'model_name': 'models/embedding-001',
    'batch_size': 25,  # Optimized for API rate limits
    'max_workers': 4,  # Balanced for performance vs rate limits
    'max_chunk_bytes': 30000,
    'retry_attempts': 3,
    'retry_delay': 0.5
}

# Chunking Configuration
CHUNKING_CONFIG = {
    'default_strategy': 'sentences',
    'target_chunks': 50,
    'overlap_ratio': 0.2,  # 20% overlap
    'max_chunk_size': 1000,  # words
    'min_chunk_size': 50   # words
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    'top_k': 5,
    'context_ratio': 0.3,
    'similarity_threshold': 0.7
}

# Generation Configuration
GENERATION_CONFIG = {
    'model_name': 'models/gemini-1.5-flash',
    'temperature': 0.1,  # Lower for more consistent answers
    'max_tokens': 1000
}

# Caching Configuration
CACHE_CONFIG = {
    'enabled': True,
    'cache_dir': 'cache',
    'max_cache_size': 1000,  # MB
    'cache_ttl': 3600  # seconds
}

# API Configuration
API_CONFIG = {
    'max_concurrent_requests': 10,
    'request_timeout': 30,
    'rate_limit_per_minute': 60
}

# Performance Monitoring
MONITORING_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',
    'performance_metrics': True
} 