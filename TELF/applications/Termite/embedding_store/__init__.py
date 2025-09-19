# Termite/embedding_store/__init__.py
import os
from .base import EmbeddingStore
from .opensearch_store import OpenSearchStore
from .milvus_store import MilvusStore

def make_store() -> EmbeddingStore:
    backend = os.getenv("EMBEDDING_STORE", "opensearch").lower()
    if backend == "milvus":
        uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        return MilvusStore(uri=uri)
    # default: OpenSearch
    host = os.getenv("OS_HOST", "localhost")
    port = int(os.getenv("OS_PORT", "9200"))
    use_ssl = os.getenv("OS_USE_SSL", "false").lower() == "true"
    username = os.getenv("OS_USERNAME")
    password = os.getenv("OS_PASSWORD")
    return OpenSearchStore(host=host, port=port, use_ssl=use_ssl,
                           username=username, password=password)