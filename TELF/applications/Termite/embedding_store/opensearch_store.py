# Termite/embedding_store/opensearch_store.py
from typing import List, Mapping, Optional, Tuple, Any
from .base import EmbeddingStore
from opensearchpy import OpenSearch, helpers
import os

def _normalize_vector(vec: Any) -> List[float]:
    """Accept list/tuple/np.ndarray or a string like '[0.1, 0.2]' → list[float]."""
    if isinstance(vec, str):
        import json, ast
        try:
            vec = json.loads(vec)
        except Exception:
            vec = ast.literal_eval(vec)

    try:
        import numpy as np
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
    except Exception:
        pass

    if not isinstance(vec, list):
        vec = list(vec)

    out = []
    for x in vec:
        if x is None:
            raise TypeError("Embedding contains None; all values must be floats.")
        out.append(float(x))
    if not out:
        raise ValueError("Embedding vector is empty.")
    return out

SPACE_MAP = {"cosine": "cosinesimil", "l2": "l2", "dot": "innerproduct"}

class OpenSearchStore(EmbeddingStore):
    def __init__(
        self,
        host: str = None,
        port: int = None,
        use_ssl: bool = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 60,
    ):
        # Allow env fallback (useful when Termite sets OS_* env vars)
        host = host or os.environ.get("OS_HOST", "localhost")
        port = int(port if port is not None else os.environ.get("OS_PORT", "9200"))
        use_ssl = bool(str(use_ssl if use_ssl is not None else os.environ.get("OS_USE_SSL", "false")).lower() == "true")

        kwargs = {
            "hosts": [{"host": host, "port": port}],
            "use_ssl": use_ssl,
            "verify_certs": False,   # fine for dev
            "http_compress": True,
            "timeout": timeout,
        }
        if username and password:
            kwargs["http_auth"] = (username, password)
        self.client = OpenSearch(**kwargs)

    # ---------- Index management ----------
    def ensure_index(
        self,
        index: str,
        dim: int,
        metric: str = "cosine",
        ef_search: Optional[int] = 128,  # OS 2.13: set at index level (per-query requires >=2.16)
        shards: int = 1,
        replicas: int = 0,
        engine: str = "nmslib",
        **kwargs
    ) -> None:
        if self.client.indices.exists(index=index):
            return

        space = SPACE_MAP.get(str(metric).lower(), "cosinesimil")
        settings = {
            "index": {
                "knn": True,
                "number_of_shards": int(shards),
                "number_of_replicas": int(replicas),
            }
        }
        if ef_search is not None:
            # OpenSearch 2.13: tune HNSW candidate breadth here
            settings["index"]["knn.algo_param.ef_search"] = int(ef_search)

        body = {
            "settings": settings,
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": int(dim),
                        "method": {
                            "name": "hnsw",
                            "engine": engine,
                            "space_type": space,
                        },
                    },
                }
            },
        }
        self.client.indices.create(index=index, body=body)

    # ---------- Write ----------
    def upsert(self, index, ids, vectors, payloads=None, refresh: bool = True) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length.")
        actions = []
        for i, vid in enumerate(ids):
            src = {
                "id": str(vid),
                "embedding": _normalize_vector(vectors[i]),
            }
            if payloads is not None and i < len(payloads) and payloads[i] is not None:
                # merge user payloads (e.g., {"text": "...", "metadata": {...}})
                src.update(payloads[i])
            actions.append({"_op_type": "index", "_index": index, "_id": str(vid), "_source": src})
        helpers.bulk(self.client, actions)
        if refresh:
            self.client.indices.refresh(index=index)

    # ---------- Read ----------
    def search(
        self,
        index: str,
        query,
        k: int = 5,
        source_fields: Optional[Any] = None,
    ) -> List[Tuple[str, float, Mapping]]:
        qvec = _normalize_vector(query)

        # _source handling: list or comma-sep string
        if source_fields is None:
            source_fields = ["id", "text", "metadata"]
        elif isinstance(source_fields, str):
            source_fields = [s.strip() for s in source_fields.split(",") if s.strip()]
        elif not isinstance(source_fields, list):
            raise TypeError(f"_source must be list[str] or comma-separated str; got {type(source_fields).__name__}")

        body = {
            "size": int(k),
            "query": {
                "knn": {
                    # OpenSearch-native syntax for 2.13:
                    "embedding": {"vector": qvec, "k": int(k)}
                }
            },
            "_source": source_fields,
        }

        resp = self.client.search(index=index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        out: List[Tuple[str, float, Mapping]] = []
        for h in hits:
            src = h.get("_source", {}) or {}
            out.append((h.get("_id", src.get("id")), float(h.get("_score", 0.0)), src))
        return out
