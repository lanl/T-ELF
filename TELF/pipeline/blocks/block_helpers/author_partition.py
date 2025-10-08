# TELF/pipeline/blocks/block_helpers/author_partition.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ast
import json
import numpy as np
import pandas as pd

UNKNOWN_COUNTRY = "Unknown"


# ----------------------------- Parsers ---------------------------------

def _parse_literal_or_json(text: str):
    """Try ast.literal_eval then JSON; return Python object or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(s)
        except Exception:
            pass
    return None


def _to_list_any(x: Any) -> List[Any]:
    """Normalize a value to a Python list (best‑effort)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        parsed = _parse_literal_or_json(x)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        # fall back to delimiter split
        sep = ";" if ";" in x else ","
        return [t.strip() for t in x.split(sep) if t.strip()]
    return [x]


def _parse_authors_field(row: pd.Series) -> List[str]:
    """
    Return a list of author IDs as strings.
    Tries SLIC → generic → S2 IDs; falls back to empty.
    """
    for cand in ("slic_author_ids", "author_ids", "s2_author_ids"):
        if cand in row and pd.notna(row[cand]):
            vals = _to_list_any(row[cand])
            return [str(v).strip() for v in vals if str(v).strip() != ""]
    return []


def _author_id_to_name_map(row: pd.Series) -> Dict[str, str]:
    """
    If both author IDs and names exist with the same length, return an id→name map.
    Otherwise return {}.
    """
    id_cols = [c for c in ("slic_author_ids", "author_ids", "s2_author_ids") if c in row.index]
    name_cols = [c for c in ("slic_authors", "authors") if c in row.index]
    for ic in id_cols:
        for nc in name_cols:
            ids = _to_list_any(row.get(ic))
            names = _to_list_any(row.get(nc))
            if len(ids) == len(names) and len(ids) > 0:
                out = {}
                for i, n in zip(ids, names):
                    sid = str(i).strip()
                    name = str(n).strip()
                    if sid:
                        out[sid] = name
                if out:
                    return out
    return {}


def _parse_affiliations_field(val: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize an affiliations field into a dict-of-dicts keyed by affiliation id (string).

    Accepts dict, list[dict], or str (Python‑literal or JSON). Returns a dict where each
    value has at least:
        { "name": str|None, "country": str, "authors": List[str] }
    """
    # 1) Convert to Python object
    if isinstance(val, (dict, list)):
        obj = val
    elif isinstance(val, str):
        obj = _parse_literal_or_json(val) or {}
    else:
        obj = {}

    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        # fabricate keys if missing
        items = [(str(i.get("id", i.get("affiliation_id", idx))), i)
                 for idx, i in enumerate(obj) if isinstance(i, dict)]
    else:
        items = []

    for k, v in items:
        if not isinstance(v, dict):
            continue
        name = v.get("name") or v.get("affiliation_name") or v.get("org") or v.get("institution")
        if isinstance(name, str) and not name.strip():
            name = None
        country = v.get("country", UNKNOWN_COUNTRY)
        if not isinstance(country, str) or not country.strip():
            country = UNKNOWN_COUNTRY

        # authors may be under various keys
        auths = v.get("authors", v.get("author_ids", []))
        auth_list = [str(a).strip() for a in _to_list_any(auths) if str(a).strip() != ""]
        out[str(k)] = {"name": name, "country": country, "authors": auth_list, **v}

    return out


def _choose_paper_id_column(df: pd.DataFrame) -> str:
    """Pick a stable paper-id column for de-duplication and counting."""
    for cand in ("eid", "s2id", "doi"):
        if cand in df.columns:
            return cand
    # synthesize one
    if "paper_id" not in df.columns:
        df["paper_id"] = np.arange(len(df)).astype(str)
    return "paper_id"


def _most_common_non_null(series: pd.Series) -> Any:
    """Return the most frequent non-null value in a Series, or np.nan."""
    s = series.dropna()
    if s.empty:
        return np.nan
    return s.value_counts().idxmax()


# ------------------------- Public API ----------------------------------

def write_top_authors_by_cluster(
    df_path: Union[str, Path],
    output_path: Union[str, Path],
    COUNTY_NAMES: Optional[Iterable[str]] = None,
    top_n: int = 10,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Build a 'top authors by cluster' table.

    Output CSV schema:
        ['cluster', 'author_id', 'author', 'affiliation_name', 'country',
         'paper_count', 'num_citations']

    Notes
    -----
    * COUNTY_NAMES is preserved for backward compatibility in the caller. If given,
      rows are filtered to only those countries.
    * Per-paper credit: each author gets 1 paper for that row (unique paper_id),
      and the row's num_citations are summed across their papers.
    """
    df_path = Path(df_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If no input, write empty CSV with expected header.
    if not df_path.is_file():
        empty = pd.DataFrame(
            columns=["cluster", "author_id", "author", "affiliation_name", "country",
                     "paper_count", "num_citations"]
        )
        empty.to_csv(out_path, index=False)
        return empty

    df = pd.read_csv(df_path)

    # Ensure basic columns
    if "cluster" not in df.columns:
        df["cluster"] = 0
    if "num_citations" not in df.columns:
        df["num_citations"] = 0

    # Choose paper id column robustly
    pid_col = _choose_paper_id_column(df)

    # Prefer SLIC affiliations, fall back
    aff_col = "slic_affiliations" if "slic_affiliations" in df.columns else (
        "affiliations" if "affiliations" in df.columns else None
    )

    # Build records
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        cluster = int(row.get("cluster", 0)) if pd.notna(row.get("cluster", np.nan)) else 0
        paper_id = str(row.get(pid_col, ""))
        citations = row.get("num_citations", 0)
        try:
            citations = float(citations)
        except Exception:
            citations = 0.0

        author_ids = _parse_authors_field(row)
        id2name = _author_id_to_name_map(row)

        # Parse affiliations + create author→(name,country) lookup
        aff_map: Dict[str, Dict[str, Any]] = {}
        if aff_col is not None and aff_col in row and pd.notna(row[aff_col]):
            aff_map = _parse_affiliations_field(row[aff_col])

        # Build reverse index: author_id -> list of (aff_name, country)
        author_to_aff: Dict[str, List[Tuple[Optional[str], str]]] = {}
        for _, info in aff_map.items():
            aff_name = info.get("name")
            country = info.get("country", UNKNOWN_COUNTRY)
            if not isinstance(country, str) or not country.strip():
                country = UNKNOWN_COUNTRY
            for aid in info.get("authors", []):
                author_to_aff.setdefault(str(aid), []).append((aff_name, country))

        # Create a record per author
        for aid in author_ids:
            aff_pairs = author_to_aff.get(aid, [])
            if aff_pairs:
                # pick most common (name,country) for this paper row
                names = pd.Series([a for a, _ in aff_pairs], dtype="object")
                cntrs = pd.Series([c for _, c in aff_pairs], dtype="object")
                aff_name = _most_common_non_null(names)
                country = _most_common_non_null(cntrs)
            else:
                aff_name = np.nan
                country = UNKNOWN_COUNTRY

            records.append(
                dict(
                    cluster=cluster,
                    paper_id=paper_id,
                    author_id=str(aid),
                    author=id2name.get(str(aid), np.nan),
                    affiliation_name=aff_name if (isinstance(aff_name, str) and aff_name.strip()) else np.nan,
                    country=country if (isinstance(country, str) and country.strip()) else UNKNOWN_COUNTRY,
                    num_citations=citations,
                )
            )

    if not records:
        empty = pd.DataFrame(
            columns=["cluster", "author_id", "author", "affiliation_name", "country",
                     "paper_count", "num_citations"]
        )
        empty.to_csv(out_path, index=False)
        return empty

    rec_df = pd.DataFrame.from_records(records)

    # Optionally filter by country list (param name preserved as in caller)
    if COUNTY_NAMES:
        counties = {str(c).strip() for c in COUNTY_NAMES if str(c).strip()}
        if counties:
            rec_df = rec_df[rec_df["country"].isin(counties)].copy()

    if rec_df.empty:
        empty = pd.DataFrame(
            columns=["cluster", "author_id", "author", "affiliation_name", "country",
                     "paper_count", "num_citations"]
        )
        empty.to_csv(out_path, index=False)
        return empty

    # Aggregate: unique paper count per (cluster, author_id) and sum citations
    agg = (
        rec_df.groupby(["cluster", "author_id"], dropna=False)
        .agg(
            paper_count=("paper_id", "nunique"),
            num_citations=("num_citations", "sum"),
            # pick most common non-null strings
            author=("author", _most_common_non_null),
            affiliation_name=("affiliation_name", _most_common_non_null),
            country=("country", _most_common_non_null),
        )
        .reset_index()
    )

    # Rank within each cluster
    agg["num_citations"] = pd.to_numeric(agg["num_citations"], errors="coerce").fillna(0.0)
    agg["paper_count"] = pd.to_numeric(agg["paper_count"], errors="coerce").fillna(0).astype(int)

    agg = agg.sort_values(["cluster", "num_citations", "paper_count"], ascending=[True, False, False])

    # Keep top_n per cluster
    top = agg.groupby("cluster", group_keys=False).head(int(top_n)).reset_index(drop=True)

    # Write and return
    top.to_csv(out_path, index=False)
    if debug:
        print(f"[author_partition] Wrote top authors by cluster → {out_path} (rows={len(top)})")
    return top
