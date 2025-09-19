# Termite/embedding_store/milvus_store.py
from typing import List, Mapping, Optional, Tuple
from .base import EmbeddingStore

from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection
import numpy as np

_METRIC_MAP = {"cosine": "COSINE", "l2": "L2", "dot": "IP"}

class MilvusStore(EmbeddingStore):
    def __init__(self, uri: str = "http://localhost:19530"):
        self.client = MilvusClient(uri=uri)

    def ensure_index(self, index: str, dim: int, metric: str = "cosine", **kwargs) -> None:
        if self.client.has_collection(index):
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields, description="Termite vectors")
        self.client.create_collection(collection_name=index, schema=schema, consistency_level="Strong")
        self.client.create_index(
            collection_name=index,
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": _METRIC_MAP.get(metric.lower(), "COSINE"), "params": {"M": 16, "efConstruction": 200}},
        )

    def upsert(
        self,
        index: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Mapping]] = None,
    ) -> None:
        rows = []
        for i, vid in enumerate(ids):
            row = {"id": vid, "embedding": vectors[i]}
            if payloads and i < len(payloads) and payloads[i] is not None:
                # expect payload to contain "text" and/or "metadata"
                row.update(payloads[i])
            rows.append(row)
        self.client.insert(collection_name=index, data=rows)

    def search(self, index: str, query: List[float], k: int = 5, **kwargs) -> List[Tuple[str, float, Mapping]]:
        res = self.client.search(
            collection_name=index,
            data=[query],
            filter=kwargs.get("filter"),
            limit=k,
            output_fields=["id", "text", "metadata"],
            search_params={"metric_type": "COSINE", "params": {"ef": kwargs.get("ef", 128)}},
        )
        hits = res[0]
        out = []
        for h in hits:
            out.append((h["entity"]["id"], float(h["distance"]), {"text": h["entity"].get("text"), "metadata": h["entity"].get("metadata")}))
        return out
