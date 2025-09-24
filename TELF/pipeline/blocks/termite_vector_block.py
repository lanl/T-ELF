# TELF/pipeline/blocks/termite_vector_index_block.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional, List

import pandas as pd
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from TELF.applications import Termite

class TermiteVectorBlock(AnimalBlock):
    """
    Mirrors test_termite_e2e.py:
      os.environ[...] for OpenSearch
      t = Termite(kg_credentials=None, verbose=True, model_name=MODEL)
      emb_map = t.compute_embeddings(df, model_name=MODEL)
      embeddings = [emb_map[i] for i in df.index]
      t.store.ensure_index(index, dim, metric='cosine')
      t.store.upsert(index, ids, embeddings, payloads=[{'text': ...}, ...])
      (optional) test search with a query string

    call_settings:
      raw_csv_path: str|Path      # default: bundle['LeafDataLabels.leaf_data_csv']
      id_column: str              # default: 'eid' (falls back to df.index if absent)
      text_column: str            # default: 'abstract'
      index_name: str             # default: 'termite_vectors'
      model_name: str             # default: 'malteos/scincl'
      metric: str                 # default: 'cosine'
      env: dict                   # optional overrides for OS_* and EMBEDDING_STORE
      test_query_text: str        # optional; if set, do a top-k search
      test_k: int                 # default: 5

    Provides:
      - '{tag}.vector_index_name'
      - '{tag}.vector_stats' (docs, dim, metric, index)
      - '{tag}.search_hits' (if test_query_text provided)
    """

    CANONICAL_NEEDS: Tuple[str, ...] = ("leaf_data_csv",)

    def __init__(self, *, needs: Sequence[str] = CANONICAL_NEEDS,
                 provides: Sequence[str] = ("vector_index_name", "vector_stats"),
                 tag: str = "TermiteVectorIndex",
                 init_settings: Optional[Dict[str, Any]] = None,
                 call_settings: Optional[Dict[str, Any]] = None,
                 verbose: bool = True, **kw: Any) -> None:
        super().__init__(needs=needs, provides=provides, tag=tag,
                         init_settings=init_settings or {}, call_settings=call_settings or {},
                         verbose=verbose, checkpoint=False, **kw)

    # ---- env like your script ----
    def _init_env(self):
        env = {
            "EMBEDDING_STORE": "opensearch",
            "OS_HOST": os.getenv("OS_HOST", "localhost"),
            "OS_PORT": os.getenv("OS_PORT", "9200"),
            "OS_USE_SSL": os.getenv("OS_USE_SSL", "false"),
        }
        overrides = self.call_settings.get("env") or {}
        env.update(overrides)
        for k, v in env.items():
            os.environ[str(k)] = str(v)

    def _load_df(self, bundle: DataBundle) -> pd.DataFrame:
        raw = self.call_settings.get("raw_csv_path") or bundle.get("LeafDataLabels.leaf_data_csv")
        if raw:
            return pd.read_csv(Path(raw).expanduser().resolve())
        # fallback to bundle df if caller wired it that way
        return bundle["df"]

    def run(self, bundle: DataBundle) -> None:
        self._init_env()
        df = self._load_df(bundle)

        model = self.call_settings.get("model_name", "malteos/scincl")
        t = Termite(kg_credentials=None, verbose=self.verbose, model_name=model)

        # embeddings aligned to df.index
        emb_map = t.compute_embeddings(df, model_name=model)
        embeddings = [emb_map[i] for i in df.index]
        if not embeddings:
            raise RuntimeError(f"[{self.tag}] No embeddings produced")
        dim = len(embeddings[0])

        index_name = self.call_settings.get("index_name", "termite_vectors")
        metric = self.call_settings.get("metric", "cosine")

        # ensure index dimension & metric
        t.store.ensure_index(index=index_name, dim=dim, metric=metric)

        # ids + payloads
        id_col = self.call_settings.get("id_column", "eid")
        ids: List[str]
        if id_col in df.columns:
            ids = df[id_col].astype(str).tolist()
        else:
            ids = df.index.astype(str).tolist()  # fallback

        text_col = self.call_settings.get("text_column", "abstract")
        if text_col not in df.columns:
            raise RuntimeError(f"[{self.tag}] text_column '{text_col}' not in DataFrame")
        payloads = [{"text": txt} for txt in df[text_col].astype(str).tolist()]

        # upsert
        t.store.upsert(index_name, ids, embeddings, payloads=payloads)

        # optional quick search
        hits_out = None
        qtxt = self.call_settings.get("test_query_text")
        if qtxt:
            # same embed path as your helper
            _qdf = pd.DataFrame({text_col: [qtxt]})
            qmap = t.compute_embeddings(_qdf, model_name=model)
            qvec = qmap[_qdf.index[0]]
            k = int(self.call_settings.get("test_k", 5))
            hits = t.store.search(index_name, qvec, k=k, source_fields="id,text")
            # normalize output for bundle
            hits_out = [{"id": _id, "score": float(score), "text": (src or {}).get("text")} for _id, score, src in hits]
            bundle[f"{self.tag}.search_hits"] = hits_out
            if self.verbose:
                print(f"[{self.tag}] Top-{k} search preview:")
                for h in hits_out:
                    print(f"  {h['score']:.4f}  id={h['id']}  text={h['text']}")

        # expose in bundle
        bundle[f"{self.tag}.vector_index_name"] = index_name
        bundle[f"{self.tag}.vector_stats"] = {
            "docs": int(len(df)),
            "dim": int(dim),
            "metric": metric,
            "index": index_name,
            "id_column": id_col,
            "text_column": text_col,
        }
        if self.verbose:
            print(f"[{self.tag}] Upserted {len(df)} vectors (dim={dim}) into '{index_name}'")
