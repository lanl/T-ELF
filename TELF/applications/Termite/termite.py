# Termite/termite.py
from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Tuple, Iterable, Any, Union

from .neo4j_termite.constants import *
from .neo4j_termite.DataInjector import InjectorNeo4j

# NEW: storage-agnostic vectorizer + store factory
#   - Vectorizer computes embeddings only
#   - make_store() returns an OpenSearch (default) or Milvus backend
from .VectorInjector import Vectorizer
from .embedding_store import make_store   # returns EmbeddingStore

import pandas as pd


class Termite:
    """
    Termite, Knowledge graph builder tool.

    Now storage-agnostic for embeddings:
      - Default backend: OpenSearch (k-NN)
      - Optional backend: Milvus (set EMBEDDING_STORE=milvus or pass embedding_backend='milvus')

    Backward-compat params kept:
      - vector_uri: used if backend == 'milvus'; ignored for OpenSearch unless you pass
                    embedding_backend_config={'host': ..., 'port': ...}
    """

    def __init__(
        self,
        kg_credentials: Optional[Tuple[str, Tuple[str, str]]] = None,
        vector_uri: str = "http://localhost:19530",
        db_nme: str = "default",
        token: Optional[str] = None,
        verbose: bool = False,
        embedding_backend: Optional[str] = None,
        embedding_backend_config: Optional[Mapping[str, Any]] = None,
        model_name: str = "malteos/scincl",
        use_gpu: bool = False,
    ):
        """
        Parameters
        ----------
        kg_credentials : tuple[str, tuple[str,str]]
            (url, (user, password)) for Neo4j
        vector_uri : str
            Back-compat. Used as Milvus URI when embedding_backend='milvus'.
        db_nme : str
            Not used by OpenSearch; kept for compatibility.
        token : str | None
            Reserved for future auth needs.
        verbose : bool
            Verbosity flag.
        embedding_backend : {'opensearch','milvus'} | None
            If None, falls back to ENV EMBEDDING_STORE or 'opensearch'.
        embedding_backend_config : Mapping
            Backend-specific knobs, e.g.
              OpenSearch: {'host':'localhost','port':9200,'use_ssl':False}
              Milvus:     {'uri':'http://localhost:19530'}
        model_name : str
            HF model to compute embeddings.
        use_gpu : bool
            Try to use CUDA for embeddings.
        """
        self.verbose = verbose

        # ---- Graph injector (unchanged) ----
        self.graph_injector = InjectorNeo4j(kg_credentials, verbose)

        # ---- Vectorizer (compute embeddings only) ----
        self.vectorizer = Vectorizer(model_name=model_name)

        # ---- Embedding store (pluggable) ----
        # precedence: explicit arg -> ENV -> default('opensearch')
        backend = (embedding_backend or os.getenv("EMBEDDING_STORE", "opensearch")).lower()
        cfg = dict(embedding_backend_config or {})

        if backend == "milvus":
            # honor legacy vector_uri if not provided in config
            cfg.setdefault("uri", vector_uri)
            os.environ.setdefault("EMBEDDING_STORE", "milvus")
        else:
            # default OpenSearch
            os.environ.setdefault("EMBEDDING_STORE", "opensearch")
            # allow simple overrides
            cfg.setdefault("host", os.getenv("OS_HOST", "localhost"))
            cfg.setdefault("port", int(os.getenv("OS_PORT", "9200")))
            cfg.setdefault("use_ssl", os.getenv("OS_USE_SSL", "false").lower() == "true")

        # make_store() reads EMBEDDING_STORE + common OS_* / MILVUS_* envs;
        # we pass overrides via env for a simple, consolidated factory.
        if backend == "milvus":
            if "uri" in cfg:
                os.environ["MILVUS_URI"] = str(cfg["uri"])
        else:
            os.environ["OS_HOST"] = str(cfg["host"])
            os.environ["OS_PORT"] = str(cfg["port"])
            os.environ["OS_USE_SSL"] = "true" if cfg.get("use_ssl") else "false"

        self.store = make_store()

        # book-keeping (optional)
        self._default_metric = "cosine"
        self._db_name = db_nme
        self._token = token
        self._model_name = model_name
        self._use_gpu = use_gpu

    # -------------------------
    # CSV → triplets (graph)
    # -------------------------

    def from_csv_to_triplets(self, csv_path: str, save_path: str, column_triplet_map: Optional[Mapping] = None):
        """
        Builds a datafile that maps the raw csv into a head-relation-tail csv
        """
        self.graph_injector.from_csv_to_triplets(csv_path, save_path, column_triplet_map)

    def make_unique_constrains(self, column_triplet_map: Optional[Mapping] = None, verbose: bool = False):
        self.graph_injector.make_unique_constrains(column_triplet_map, verbose)

    def iterate_csv_triplets_into_graph(self, triplets_path: str, start_from: int = 0, args: Mapping = {}):
        """
        Iterates parsed data to call the injection to graph function.
        Returns list of failed indices.
        """
        self.graph_injector.iterate_csv_triplets_into_graph(triplets_path, start_from, args)

    def update_database_multithreaded(self, triplets_path: str, start_from: int = 0, shuffle_rows: bool = True):
        """
        Iterates parsed data to call the injection to graph function (multithreaded).
        """
        self.graph_injector.update_database_multithreaded(triplets_path, start_from, shuffle_rows)

    # -------------------------
    # VECTOR STORE (generic)
    # -------------------------

    def make_vector_schema(self, collection_name: str = "slic", schema: Optional[Mapping] = None, dim: Optional[int] = None, metric: Optional[str] = None):
        """
        Creates an index/collection in the selected vector backend.

        For OpenSearch, 'schema' is ignored; we derive a simple mapping with a single
        knn_vector field named 'embedding', plus 'id', 'text', 'metadata'.

        Parameters
        ----------
        collection_name : str
        schema : dict | None
            (Kept for compatibility; Milvus users can pass params if they want.)
        dim : int | None
            Dimension of the embedding. If None, we try to infer from 'schema',
            otherwise you must supply it before inserting data.
        metric : str | None
            'cosine' (default), 'l2', or 'dot'
        """
        metric = (metric or self._default_metric).lower()
        if dim is None:
            # Try to infer from schema (Milvus-style) else refuse early
            if isinstance(schema, dict) and "dim" in schema:
                dim = int(schema["dim"])
            else:
                raise ValueError("make_vector_schema requires 'dim' (embedding dimension).")
        self.store.ensure_index(index=collection_name, dim=dim, metric=metric)

    def inject_vectors(self, collection_name: str, data: List[Mapping[str, Any]]):
        """
        Insert / upsert vectors into the selected backend.

        Expected 'data' shape (per item):
            {
              'id': str,
              'embedding': List[float],     # length = dim
              # optional user payload:
              'text': str,
              'metadata': Mapping[str, Any]
            }
        """
        ids: List[str] = []
        vecs: List[List[float]] = []
        payloads: List[Mapping[str, Any]] = []

        for row in data:
            if "id" not in row or "embedding" not in row:
                raise ValueError("Each data row must contain 'id' and 'embedding'.")
            ids.append(str(row["id"]))
            vecs.append(list(row["embedding"]))
            payloads.append({k: v for k, v in row.items() if k not in ("id", "embedding")})

        self.store.upsert(index=collection_name, ids=ids, vectors=vecs, payloads=payloads)

    # -------------------------
    # HELPERS to shape data
    # -------------------------

    def df_to_data(
        self,
        embeddings: List[List[float]],
        df_path: str,
        columns_collection_map: Mapping[str, str],
        id_col: Optional[str] = None,
        text_col: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Converts a CSV into a list of upsertable vector rows.

        Parameters
        ----------
        embeddings : list[list[float]]
            Embeddings in the same row-order as the CSV.
        df_path : str
            Path to CSV.
        columns_collection_map : dict
            Mapping of wanted CSV columns → payload keys (e.g., {"abstract":"text","paper_id":"id"}).
            If 'id' or 'text' are not provided here, we will fallback to id_col/text_col args.
        id_col, text_col : Optional[str]
            Fallbacks for the identifier and text payload.

        Returns
        -------
        List[dict]
            Each dict at least has {'id', 'embedding'} plus any mapped payload fields.
        """
        df = pd.read_csv(df_path)
        if len(df) != len(embeddings):
            raise ValueError(f"embeddings length ({len(embeddings)}) != rows in CSV ({len(df)})")

        # Determine id/text columns
        mapped_id = next((src for src, dst in columns_collection_map.items() if dst == "id"), None) or id_col
        mapped_text = next((src for src, dst in columns_collection_map.items() if dst == "text"), None) or text_col

        data: List[Dict[str, Any]] = []
        for i, row in df.iterrows():
            payload: Dict[str, Any] = {}

            # Map requested columns
            for src, dst in columns_collection_map.items():
                if src in df.columns:
                    payload[dst] = row[src]

            # Ensure id/text presence if desired
            rid = str(row[mapped_id]) if mapped_id and mapped_id in df.columns else str(i)
            if "id" not in payload:
                payload["id"] = rid

            if mapped_text and mapped_text in df.columns and "text" not in payload:
                payload["text"] = row[mapped_text]

            data.append({"id": payload.pop("id"), "embedding": embeddings[i], **payload})

        return data

    # -------------------------
    # EMBEDDING COMPUTATION
    # -------------------------

    def compute_embeddings(
        self,
        df: "pd.DataFrame",
        model_name: str = "SCINCL",
        use_gpu: bool = False,
        text_column: Optional[str] = None,
    ) -> Dict[Any, List[float]]:
        """
        Computes embeddings for a pandas DataFrame.

        Assumes there is a textual column to embed. If `text_column` is None,
        we try 'text', then the first object/string dtype column.

        Returns
        -------
        dict: {row_index -> embedding_vector}
        """
        # choose column
        col = text_column
        if col is None:
            if "text" in df.columns:
                col = "text"
            else:
                # pick first object-dtype column as a best-effort fallback
                obj_cols = [c for c in df.columns if df[c].dtype == object]
                if not obj_cols:
                    raise ValueError("No text column found. Provide 'text_column'.")
                col = obj_cols[0]

        texts: List[str] = df[col].astype(str).tolist()

        # swap model if caller passed one
        if model_name and model_name != self._model_name:
            self.vectorizer = Vectorizer(model_name=model_name)
            self._model_name = model_name

        # note: Vectorizer selects CUDA automatically if available; we keep the flag for API parity
        embs = self.vectorizer.encode(texts)

        # map back to index
        return {idx: embs[i] for i, idx in enumerate(df.index.tolist())}

    # -------------------------
    # OPTIONAL: convenience search
    # -------------------------

    def search_vectors(
        self,
        collection_name: str,
        query_vec: List[float],
        k: int = 5,
        **kwargs,
    ) -> List[Tuple[str, float, Mapping]]:
        """
        Simple k-NN search wrapper to the active backend.
        Returns a list of (id, score, payload).
        """
        return self.store.search(index=collection_name, query=query_vec, k=k, **kwargs)
