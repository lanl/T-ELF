from __future__ import annotations
# TELF/pipeline/blocks/beaver_codependency_matrix_block.py

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pickle
import scipy.sparse as sp
import sparse  # pydata/sparse

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# Ensure Beaver.coauthor_tensor is patched to a robust version on import
# from ...pre_processing import Beaver

# TELF/pre_processing/Beaver/monkey_patch_coauthor_tensor.py
"""
Monkey-patch Beaver.coauthor_tensor to be robust to:
- n_jobs <= 0 (uses CPU count)
- empty / missing authors (writes an empty but valid tensor)
- missing 'year' column (uses 0)
Also saves authors/time index maps for downstream consumers.
"""


import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sparse  # pydata/sparse

from ...pre_processing import Beaver


def _safe_coauthor_tensor(
    self: Beaver,
    *,
    dataset: pd.DataFrame,
    target_columns: List[str],
    split_authors_with: str = ";",
    verbose: int = 0,
    save_path: str | None = None,
    n_nodes: int | None = None,
    n_jobs: int = 1,
    joblib_backend: str | None = None,
    authors_idx_map: Dict[str, int] | None = None,
    time_idx_map: Dict[int, int] | None = None,
    return_object: bool = False,
    output_mode: str | None = None,
):
    auth_col, time_col = target_columns
    df = dataset.copy()

    if time_col not in df.columns:
        df[time_col] = 0

    # Parse authors per doc
    auth_series = (
        df[auth_col].fillna("")
        .astype(str)
        .str.split(split_authors_with)
        .apply(lambda lst: [a.strip() for a in lst if a and a.strip()])
    )
    times = df[time_col].fillna(0).astype(int).tolist()

    # Build indices
    if authors_idx_map is None:
        unique_authors = sorted({a for lst in auth_series.tolist() for a in lst})
        a2i: Dict[str, int] = {a: i for i, a in enumerate(unique_authors)}
    else:
        a2i = dict(authors_idx_map)

    if time_idx_map is None:
        unique_times = sorted(set(int(t) for t in times))
        if not unique_times:
            unique_times = [0]
        t2i: Dict[int, int] = {t: i for i, t in enumerate(unique_times)}
    else:
        t2i = dict(time_idx_map)

    A = len(a2i)
    T = len(t2i) if t2i else 1

    # Early exit: no authors → empty but valid 3D tensor (0 x 0 x max(1,T))
    if A == 0:
        coo = sparse.COO(np.zeros((0, 0, T), dtype=np.float32))
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            sparse.save_npz(os.path.join(save_path, "coauthor.npz"), coo)
            with open(os.path.join(save_path, "authors_idx_map.p"), "wb") as f:
                pickle.dump(a2i, f)
            with open(os.path.join(save_path, "time_idx_map.p"), "wb") as f:
                pickle.dump(t2i, f)
        return coo if return_object else None

    # Build weighted undirected pairs per time
    from collections import Counter

    weight = Counter()
    for lst, t in zip(auth_series.tolist(), times):
        idxs = [a2i[a] for a in lst if a in a2i]
        if len(idxs) < 2:
            continue
        ti = t2i.get(int(t), next(iter(t2i.values())) if t2i else 0)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                u, v = idxs[i], idxs[j]
                weight[(u, v, ti)] += 1
                weight[(v, u, ti)] += 1  # undirected

    if weight:
        coords = np.array(list(zip(*weight.keys())))
        data = np.array(list(weight.values()), dtype=np.float32)
        coo = sparse.COO(coords, data, shape=(A, A, T))
    else:
        coo = sparse.COO(np.zeros((A, A, T), dtype=np.float32))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        sparse.save_npz(os.path.join(save_path, "coauthor.npz"), coo)
        with open(os.path.join(save_path, "authors_idx_map.p"), "wb") as f:
            pickle.dump(a2i, f)
        with open(os.path.join(save_path, "time_idx_map.p"), "wb") as f:
            pickle.dump(t2i, f)

    return coo if return_object else None


# Apply the patch at import time
Beaver.coauthor_tensor = _safe_coauthor_tensor  # type: ignore[misc]


class CodependencyMatrixBlock(AnimalBlock):
    """
    Build a (flattened) co-dependency matrix from a column of semicolon-separated ids + 'year'.

    needs:    ['df']
    provides: ['X', 'node_ids']
    """

    def __init__(
        self,
        *,
        col: str,
        needs: Sequence[str] = ("df",),
        provides: Sequence[str] = ("X", "node_ids"),
        tag: str = "BeaverCodependencyMatrix",
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        self.col = col
        default_init: Dict[str, Any] = {}
        default_call: Dict[str, Any] = {"split_authors_with": ";", "n_jobs": 1}
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings={**default_init, **(init_settings or {})},
            call_settings={**default_call, **(call_settings or {})},
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        raw = bundle[self.needs[0]]
        df = self.load_path(raw) if isinstance(raw, (str, Path)) else raw

        # Ensure 'year' exists (Beaver expects it for the 3-mode tensor)
        if "year" not in df.columns:
            df = df.copy()
            df["year"] = 0

        out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run Beaver to materialize a 3D co-author tensor (A x A x T)
        beaver = Beaver()
        cfg = dict(self.call_settings)
        cfg.update(
            {
                "dataset": df,
                "target_columns": [self.col, "year"],
                "save_path": str(out_dir),
            }
        )
        beaver.coauthor_tensor(**cfg)

        # Load the tensor and flatten across time
        coo3: "sparse.COO" = sparse.load_npz(out_dir / "coauthor.npz")
        # flatten T mode → 2D A x A
        coo2 = coo3.sum(axis=2)

        # Convert to scipy.sparse (CSR)
        if hasattr(coo2, "coords") and hasattr(coo2, "data"):
            rows, cols = coo2.coords[0], coo2.coords[1]
            X = sp.csr_matrix((coo2.data, (rows, cols)), shape=coo2.shape)
        else:
            # Fallback
            X = sp.csr_matrix(np.asarray(coo2))

        # Node id order from Beaver's authors_idx_map if present
        a_map_path = out_dir / "authors_idx_map.p"
        if a_map_path.exists():
            with open(a_map_path, "rb") as f:
                a2i: Dict[str, int] = pickle.load(f)
            node_ids = [None] * len(a2i)
            for a, idx in a2i.items():
                node_ids[idx] = a
        else:
            # Fallback: derive from df (order might differ from Beaver)
            ids = (
                df[self.col]
                .dropna()
                .astype(str)
                .str.split(cfg.get("split_authors_with", ";"))
                .explode()
                .str.strip()
            )
            ids = ids.loc[ids != ""].unique().tolist()
            node_ids = sorted(set(ids))

        # Publish outputs (top-level keys; WolfBlock expects this)
        bundle[self.provides[0]] = X
        bundle[self.provides[1]] = node_ids
