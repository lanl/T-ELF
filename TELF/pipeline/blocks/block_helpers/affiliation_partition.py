# TELF/pipeline/blocks/block_helpers/affiliation_partition.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ast
import json
import numpy as np
import pandas as pd

UNKNOWN_COUNTRY = "Unknown"


def _parse_affiliations_field(val: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize an affiliations field into a dict-of-dicts keyed by affiliation id.

    Accepts:
      - dict (already keyed by id) with nested dicts
      - list[dict] (each item is an affiliation)
      - string (Python-literal or JSON) representing either of the above
      - else -> empty dict

    Ensures each nested dict has keys:
      - 'name' (str)
      - 'country' (str, default 'Unknown')
    """
    # 1) Convert to Python object
    obj = None
    if isinstance(val, (dict, list)):
        obj = val
    elif isinstance(val, str):
        s = val.strip()
        if s:
            # Try Python literal first (TELF often saves repr strings)
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(s)
                    if isinstance(parsed, (dict, list)):
                        obj = parsed
                        break
                except Exception:
                    pass
        if obj is None:
            obj = {}
    else:
        obj = {}

    # 2) Canonicalize to dict-of-dicts keyed by id (string keys)
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(v, dict):
                continue
            name = v.get("name")
            country = v.get("country", UNKNOWN_COUNTRY)
            if name is None or (isinstance(name, str) and name.strip() == ""):
                # try fallbacks (some loaders store name under different keys)
                name = v.get("affiliation_name") or v.get("org") or v.get("institution") or None
            if not isinstance(country, str) or not country.strip():
                country = UNKNOWN_COUNTRY
            out[str(k)] = {**v, "name": name, "country": country}
        return out

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            key = str(item.get("id", item.get("affiliation_id", i)))
            name = item.get("name") or item.get("affiliation_name") or item.get("org") or item.get("institution")
            country = item.get("country", UNKNOWN_COUNTRY)
            if not isinstance(country, str) or not country.strip():
                country = UNKNOWN_COUNTRY
            out[key] = {**item, "name": name, "country": country}
        return out

    return {}


def _pairs_from_affiliations(val: Any) -> List[Tuple[Optional[str], str]]:
    """
    Produce a list of (affiliation_name, country) pairs from a raw affiliations field.
    Ensures each pair has exactly two elements; missing pieces are filled with defaults.
    """
    norm = _parse_affiliations_field(val)
    pairs: List[Tuple[Optional[str], str]] = []
    for _, rec in norm.items():
        name = rec.get("name")
        if isinstance(name, str):
            name = name.strip() or None
        country = rec.get("country", UNKNOWN_COUNTRY)
        if not isinstance(country, str) or not country.strip():
            country = UNKNOWN_COUNTRY
        pairs.append((name, country))
    return pairs


def _resolve_paper_id_column(df: pd.DataFrame) -> str:
    """
    Choose a robust paper-id column for grouping:
    priority: 'eid' -> 's2id' -> 'doi' -> synthetic 'paper_id'
    Returns the *name* of the column (and creates synthetic if needed).
    """
    for cand in ("eid", "s2id", "doi"):
        if cand in df.columns:
            return cand
    df = df.reset_index(drop=True)
    df["paper_id"] = df.index.astype(str)
    return "paper_id"


def generate_top_affiliations_with_country(
    df_path: Union[str, Path],
    affils_output_path: Union[str, Path],
    min_total_papers: int = 1,
    country_filter: Optional[str] = None,
    partition_by_year: bool = False,
    per_year_output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Read the (already processed) pipeline CSV and emit a table of top affiliations
    with countries, optionally partitioned by year.

    Output schema (affils_output_path):
        ['affiliation_name', 'country', 'year', 'paper_count']

    Parameters
    ----------
    df_path : path to the dataframe (CSV) produced upstream
    affils_output_path : where to write the aggregated CSV
    min_total_papers : keep only affiliations with >= this many papers overall
    country_filter : if given, keep only rows matching this country (case-sensitive)
    partition_by_year : if True, also write per-year CSVs under per_year_output_dir
    per_year_output_dir : base directory for per-year outputs; if None and
                          partition_by_year is True, defaults to affils_output_path.parent / 'by_year'
    """
    df_path = Path(df_path)
    out_path = Path(affils_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not df_path.is_file():
        # nothing to do; write an empty file with the expected header
        empty = pd.DataFrame(columns=["affiliation_name", "country", "year", "paper_count"])
        empty.to_csv(out_path, index=False)
        return

    df = pd.read_csv(df_path)

    # pick affiliations column (prefer SLIC)
    aff_col = "slic_affiliations" if "slic_affiliations" in df.columns else (
        "affiliations" if "affiliations" in df.columns else None
    )
    if aff_col is None:
        # no affiliations at all -> write empty
        empty = pd.DataFrame(columns=["affiliation_name", "country", "year", "paper_count"])
        empty.to_csv(out_path, index=False)
        return

    # ensure year column
    if "year" not in df.columns:
        df["year"] = 0
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    # robust paper id
    pid_col = _resolve_paper_id_column(df)
    if pid_col not in df.columns:
        # _resolve_paper_id_column may have added a synthetic col; ensure present
        df = df.reset_index(drop=True)
        df["paper_id"] = df.index.astype(str)
        pid_col = "paper_id"

    # Build normalized pairs per row and explode
    df = df[[pid_col, "year", aff_col]].copy()
    df["affil_pairs"] = df[aff_col].apply(_pairs_from_affiliations)

    # explode to one row per (paper, affiliation)
    exploded = df.explode("affil_pairs")

    # Normalize pairs so every row is a *2-tuple* (name, country)
    def _safe_pair(x: Any) -> Tuple[Optional[str], str]:
        if isinstance(x, (list, tuple)):
            if len(x) >= 2:
                name, country = x[0], x[1]
            elif len(x) == 1:
                name, country = x[0], UNKNOWN_COUNTRY
            else:
                name, country = None, UNKNOWN_COUNTRY
        elif isinstance(x, dict):
            name = x.get("name")
            country = x.get("country", UNKNOWN_COUNTRY)
        else:
            name, country = None, UNKNOWN_COUNTRY

        if isinstance(name, str):
            name = name.strip() or None
        if not isinstance(country, str) or not country.strip():
            country = UNKNOWN_COUNTRY
        return (name, country)

    exploded["affil_pairs"] = exploded["affil_pairs"].apply(_safe_pair)
    # Now guaranteed to be 2 columns
    exploded[["affiliation_name", "country"]] = pd.DataFrame(
        exploded["affil_pairs"].tolist(), index=exploded.index
    )

    # Drop rows where affiliation name is missing after normalization
    exploded = exploded.dropna(subset=["affiliation_name"]).copy()

    # Optional country filter
    if country_filter:
        exploded = exploded.loc[exploded["country"] == str(country_filter)].copy()

    if exploded.empty:
        out = pd.DataFrame(columns=["affiliation_name", "country", "year", "paper_count"])
        out.to_csv(out_path, index=False)
        return

    # Unique by (paper, affiliation) to avoid double-counting the same affiliation within a paper
    exploded = exploded[[pid_col, "year", "affiliation_name", "country"]].drop_duplicates()

    # Compute total papers per affiliation across all years (for thresholding)
    totals = (
        exploded.groupby(["affiliation_name", "country"])[pid_col]
        .nunique()
        .reset_index(name="paper_count_total")
    )

    # Keep only affiliations that meet the threshold
    keep = totals.loc[totals["paper_count_total"] >= int(min_total_papers), ["affiliation_name", "country"]]
    if keep.empty:
        out = pd.DataFrame(columns=["affiliation_name", "country", "year", "paper_count"])
        out.to_csv(out_path, index=False)
        return

    # Join to filter
    exploded = exploded.merge(keep, on=["affiliation_name", "country"], how="inner")

    # Aggregate by year
    by_year = (
        exploded.groupby(["affiliation_name", "country", "year"])[pid_col]
        .nunique()
        .reset_index(name="paper_count")
        .sort_values(["paper_count", "year"], ascending=[False, True])
        .reset_index(drop=True)
    )

    by_year.to_csv(out_path, index=False)

    # Optionally write per-year partitions
    if partition_by_year:
        base = Path(per_year_output_dir) if per_year_output_dir else out_path.parent / "by_year"
        base.mkdir(parents=True, exist_ok=True)
        for yr, sub in by_year.groupby("year"):
            sub_path = base / f"affiliations_{yr}.csv"
            sub.sort_values("paper_count", ascending=False).to_csv(sub_path, index=False)
