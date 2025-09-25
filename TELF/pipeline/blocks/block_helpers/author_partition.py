import ast
import json
import re
import pandas as pd

def write_top_authors_by_cluster(
    df_path: str,
    output_path: str,
    COUNTY_NAMES=None,   # keep name to preserve identical print message
    top_n: int = 10,
    debug: bool = False  # optional: prints row counts at key steps
):
    """
    Runs the same pipeline but with robust parsing, flexible author extraction,
    normalized country filtering, and sensible fallbacks.
    Keeps columns, encoding, and print message identical to your original.
    """
    def _debug(msg):
        if debug:
            print(msg)

    # ---------- helpers ----------
    _split_re = re.compile(r"[;,|]\s*")

    def _safe_eval_aff(s):
        """Parse affiliations cell to dict-like; return {} on any issue."""
        if isinstance(s, dict):
            return s
        if isinstance(s, list):
            return s
        if pd.isna(s):
            return {}
        txt = str(s).strip()
        if not txt:
            return {}
        # Try JSON first (handles true/false/null)
        try:
            return json.loads(txt)
        except Exception:
            pass
        # Fallback to Python literal
        try:
            return ast.literal_eval(txt)
        except Exception:
            return {}

    def _listify_author_ids(value):
        """Return a list[str] of author IDs from various shapes (list/json/csv)."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        # Already list-like?
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value if str(x).strip()]
        s = str(value).strip()
        if not s:
            return []
        # Try JSON array
        try:
            arr = json.loads(s)
            if isinstance(arr, (list, tuple, set)):
                return [str(x) for x in arr if str(x).strip()]
        except Exception:
            pass
        # Delimited fallback
        parts = _split_re.split(s)
        return [p for p in parts if p]

    def _authors_from_affinfo(x, row_level_authors):
        """
        Extract authors from an affiliation 'info' dict, falling back to row-level authors
        if none are present for that affiliation.
        """
        if not isinstance(x, dict):
            return row_level_authors
        for k in ("authors", "author_ids", "authorId", "author_id", "authorIds"):
            if k in x and x[k] is not None:
                ids_ = _listify_author_ids(x[k])
                if ids_:
                    return ids_
        # fallback: use the paper's row-level authors (prevents empty explode)
        return row_level_authors

    def _normalize_aff_dict(aff):
        """
        Accept dict-of-dicts OR list-of-dicts and return dict[str_id] -> dict(info).
        """
        if isinstance(aff, dict):
            return aff
        if isinstance(aff, list):
            out = {}
            for item in aff:
                if isinstance(item, dict):
                    aff_id = (item.get("id") or item.get("affiliation_id") or item.get("affiliationId") or
                              item.get("grid") or item.get("ror") or item.get("name") or "unknown")
                    out[str(aff_id)] = item
            return out
        return {}

    # ---------- load & standardize ----------
    df = pd.read_csv(df_path)

    if 'cluster' not in df.columns:
        if 'Graph_Name' in df.columns:
            df = df.copy()
            df['cluster'] = df['Graph_Name']
        else:
            raise KeyError("Expected a 'cluster' column; neither 'cluster' nor 'Graph_Name' found.")

    # Build author_name_map (author_id ↔ author_name) from row-level columns
    if not {'author_ids', 'authors'}.issubset(df.columns):
        # Create empty map to avoid KeyError; we'll still count by ID
        author_name_map = pd.DataFrame(columns=['author_id', 'author_name'])
    else:
        author_name_map = (
            df[['author_ids', 'authors']]
            .assign(
                author_ids=lambda d: d['author_ids'].astype(str).apply(_listify_author_ids),
                authors=lambda d: d['authors'].astype(str).apply(_listify_author_ids),
            )
            .explode(['author_ids', 'authors'])
            .rename(columns={'author_ids': 'author_id', 'authors': 'author_name'})
            .assign(author_id=lambda d: d['author_id'].astype(str))
            .drop_duplicates(['author_id', 'author_name'])
        )

    # ---------- explode affiliations -> (cluster, affiliation, country, author_id) ----------
    # Carry row-level author_ids to allow fallback when an affiliation lacks per-affiliation authors
    base_cols = ['cluster', 'affiliations']
    if 'author_ids' in df.columns:
        base_cols.append('author_ids')
    if 'authors' in df.columns:
        base_cols.append('authors')

    tmp = (
        df[base_cols]
        .assign(
            row_authors=lambda d: d.get('author_ids', pd.Series([None]*len(d))).apply(_listify_author_ids)
        )
        .assign(aff_raw=lambda d: d['affiliations'].map(_safe_eval_aff))
        .assign(aff_dict=lambda d: d['aff_raw'].map(_normalize_aff_dict))
        .assign(aff_items=lambda d: d['aff_dict'].map(lambda x: list(x.items())))
        .explode('aff_items', ignore_index=True)
    )

    # If nothing at all parsed, make a single "Unknown" slot per row so we can still count
    if tmp['aff_items'].isna().all():
        tmp = (
            df[['cluster']].copy()
            .assign(
                row_authors=lambda d: df.get('author_ids', pd.Series([None]*len(df))).apply(_listify_author_ids),
                aff_items=[('unknown', {'name': 'Unknown', 'country': 'unknown'})] * len(df)
            )
            .explode('aff_items', ignore_index=True)
        )

    auth_aff = (
        tmp
        .dropna(subset=['aff_items'])
        .assign(
            affiliation_id=lambda d: d['aff_items'].map(lambda p: p[0]),
            aff_info=lambda d: d['aff_items'].map(lambda p: p[1]),
        )
        .assign(
            affiliation_name=lambda d: d['aff_info'].map(lambda x: x.get('name') if isinstance(x, dict) else None),
            country_raw=lambda d: d['aff_info'].map(lambda x: x.get('country') if isinstance(x, dict) else None),
        )
    )

    # Author IDs from affiliation info, with fallback to row-level authors
    if 'row_authors' not in auth_aff.columns:
        auth_aff['row_authors'] = [[]] * len(auth_aff)
    auth_aff = auth_aff.assign(
        author_ids_list=lambda d: d.apply(
            lambda r: _authors_from_affinfo(r['aff_info'], r['row_authors']), axis=1
        )
    ).explode('author_ids_list', ignore_index=True)

    # If *still* empty, bail out with a zero-row CSV but keep the same columns
    if auth_aff.empty:
        out = pd.DataFrame(columns=[
            'cluster', 'rank', 'author_name', 'author_id', 'affiliation_name', 'country'
        ])
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(out)} rows (authors in {COUNTY_NAMES}) to {output_path}")
        return out

    auth_aff = auth_aff.assign(
        author_id=lambda d: d['author_ids_list'].astype(str),
        country=lambda d: d['country_raw'].astype(str).str.strip().replace({'': 'unknown', 'None': 'unknown', 'nan': 'unknown'})
    ).merge(author_name_map, on='author_id', how='left')

    _debug(f"rows after aff explode: {len(auth_aff)}")

    # ---------- optional country filter (normalized) ----------
    if COUNTY_NAMES:
        # Normalize both sides (casefold + strip) and handle common US aliases
        aliases = {
            'us': {'us', 'usa', 'u.s.', 'u.s.a.', 'united states', 'united states of america', 'u.s.a'},
        }
        def _norm_country(s):
            s = (s or "").strip().casefold()
            if s in aliases['us']:
                return 'united states'
            return s

        want = { _norm_country(x) for x in COUNTY_NAMES }
        auth_aff = auth_aff.assign(_country_norm=auth_aff['country'].map(_norm_country))
        before = len(auth_aff)
        auth_aff = auth_aff[auth_aff['_country_norm'].isin(want)].copy()
        auth_aff.drop(columns=['_country_norm'], inplace=True)
        _debug(f"country filter kept {len(auth_aff)} / {before} rows")

    # ---------- counts ----------
    counts = (
        auth_aff
        .groupby(['cluster', 'author_id', 'author_name', 'affiliation_name', 'country'], dropna=False)
        .size()
        .reset_index(name='paper_count')
    )

    if counts.empty:
        # Write an empty CSV with the right columns
        out = pd.DataFrame(columns=[
            'cluster', 'rank', 'author_name', 'author_id', 'affiliation_name', 'country'
        ])
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(out)} rows (authors in {COUNTY_NAMES}) to {output_path}")
        return out

    # ---------- top N per cluster ----------
    top_authors = (
        counts
        .groupby('cluster', group_keys=False)
        .apply(lambda g: g.nlargest(top_n, 'paper_count'))
        .reset_index(drop=True)
    )
    top_authors['rank'] = (
        top_authors
        .groupby('cluster')['paper_count']
        .rank(method='first', ascending=False)
        .astype(int)
    )

    # ---------- pivot cross-cluster ----------
    pivot = (
        counts
        .pivot_table(
            index=['author_id', 'author_name', 'affiliation_name', 'country'],
            columns='cluster',
            values='paper_count',
            aggfunc='sum',
            fill_value=0,
        )
        .reindex(sorted(counts['cluster'].unique()), axis=1)
        .astype(int)
        .reset_index()
    )

    # ---------- merge & write ----------
    result = top_authors.merge(
        pivot,
        on=['author_id', 'author_name', 'affiliation_name', 'country'],
        how='left'
    )

    cluster_cols = [c for c in pivot.columns if c not in ['author_id', 'author_name', 'affiliation_name', 'country']]
    cols = [
        'cluster', 'rank',
        'author_name', 'author_id',
        'affiliation_name', 'country',
    ] + cluster_cols

    out = result[cols]
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(result)} rows (authors in {COUNTY_NAMES}) to {output_path}")
    return out
