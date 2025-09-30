# TELF/pipeline/blocks/termite_neo4j_block.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional, List
import pandas as pd
from copy import deepcopy
import ast
import json
import re

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# --- Termite + constants ---
from TELF.applications import Termite
from TELF.applications.Termite.neo4j_termite import (
    ENTITY, RETURN_TYPE, ATTRIBUTES, ET, YEAR_TYPE, FROM_COL, DOCUMENT_TYPE,
    ROW_INDEX, ATTR_COL, ATTR_NAME, TT, R, DOCUMENT_YEAR_RELATION, HT,
    AUTHOR_DOCUMENT_RELATION, AUTHOR_ID_TYPE, EXTRACT_H, DOCUMENT_CITES_RELATION,
    DOCUMENT_CITED_RELATION, DOCUMENT_TYPE_SCOPUS, EXTRACT_T, PAIRING, HEAD_TO_MANY,
    TOPIC_TYPE, MAKE_ID_UNIQUE, KEYWORD_TYPE, INDEX_PAIRING,
    DOCUMENT_PUBLISHER_RELATION, PUBLISHER, AFFILIATION_IDENTIFIER_TYPE, COUNTRY_TYPE,
    CATEGORY, ACRONYM, RETREIVAL, ATTR_FUNC, ARGS, DOCUMENT_AFFILITATION_RELATION,
    AFFILIATION_COUNTRY_RELATION, DOCUMENT_CATEGORY_RELATION, DOCUMENT_ACRONYM_RELATION
)

# --------------------------- NER labels
NER_LABELS = [
    "ORG", "PERSON", "GPE", "NORP", "FAC", "LOC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
]

# ======================================================================================
# Helpers (robust access & parsing)
# ======================================================================================

def _get(row, key, default=None):
    """Accessor that works for Series/dict/object."""
    try:
        if key in row:
            return row[key]
    except Exception:
        pass
    if hasattr(row, key):
        return getattr(row, key)
    if hasattr(row, "get"):
        try:
            return row.get(key, default)
        except Exception:
            pass
    return default

def _safe_get(row, key, default=None):
    return _get(row, key, default)

def list_split_no_attrs(data, split_with=';'):
    out = []
    if isinstance(data, str):
        for v in data.split(split_with):
            e = deepcopy(RETURN_TYPE)
            e[ENTITY] = v.strip()
            out.append(e)
        return out
    return [deepcopy(RETURN_TYPE)]

def get_cites(args):
    return list_split_no_attrs(args['data'].citations)

def get_cited(args):
    return list_split_no_attrs(args['data'].references)

# --------- Generic author extractor (uses detected column names) ----------
def make_get_authors_ID_from(ids_col: str, names_col: Optional[str]):
    def _fn(args, _ids_col=ids_col, _names_col=names_col):
        row = args.get('data', None)
        if row is None:
            return [deepcopy(RETURN_TYPE)]
        data = _safe_get(row, _ids_col, None)
        if not isinstance(data, str):
            return [deepcopy(RETURN_TYPE)]
        ids = [t.strip() for t in data.split(';') if t.strip()]
        names = []
        if _names_col:
            raw_names = _safe_get(row, _names_col, "")
            if isinstance(raw_names, str) and raw_names.strip():
                names = [t.strip() for t in raw_names.split(';')]
        out = []
        for i, name in zip(ids, names + [""] * max(0, len(ids) - len(names))):
            e = deepcopy(RETURN_TYPE)
            e[ENTITY] = i
            if name:
                e[ATTRIBUTES] = [('name', name)]
            out.append(e)
        return out or [deepcopy(RETURN_TYPE)]
    return _fn

# --------- Affiliation parsing consistent with Peacock’s normalization ----------
def _aff_to_dict(cell) -> Dict[str, Dict[str, Any]]:
    """
    Accept dict/list/JSON/string; return dict keyed by affiliation id.
    Ensures 'authors' (list) & 'country' (str) are present in each value.
    """
    # parse to python object
    if isinstance(cell, (dict, list)):
        obj = cell
    elif isinstance(cell, str):
        s = cell.strip()
        if not s or s.lower() == 'nan':
            obj = {}
        else:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                try:
                    obj = json.loads(s)
                except Exception:
                    obj = {}
    else:
        obj = {}

    # canonicalize
    if isinstance(obj, list):
        out = {}
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            key = str(item.get("id", item.get("affiliation_id", i)))
            out[key] = {
                **item,
                "name": item.get("name", item.get("affiliation_name")),
                "authors": item.get("authors", item.get("author_ids", [])) or [],
                "country": item.get("country", "Unknown"),
            }
        return out
    elif isinstance(obj, dict):
        out = {}
        for k, val in obj.items():
            if not isinstance(val, dict):
                continue
            val.setdefault("name", val.get("affiliation_name"))
            val.setdefault("authors", val.get("author_ids", []))
            val.setdefault("country", "Unknown")
            out[str(k)] = val
        return out
    return {}

def make_get_affiliations_from(aff_col: str):
    def _fn(args, _aff_col=aff_col):
        row = args.get('data', None)
        if row is None:
            return [deepcopy(RETURN_TYPE)]
        raw = _safe_get(row, _aff_col, None)
        d = _aff_to_dict(raw)
        out = []
        for k, v in d.items():
            e = deepcopy(RETURN_TYPE)
            e[ENTITY] = k
            e[ATTRIBUTES] = [('name', v.get('name'))]
            out.append(e)
        return out or [deepcopy(RETURN_TYPE)]
    return _fn

def make_get_countries_from(aff_col: str):
    def _fn(args, _aff_col=aff_col):
        row = args.get('data', None)
        if row is None:
            return [deepcopy(RETURN_TYPE)]
        raw = _safe_get(row, _aff_col, None)
        d = _aff_to_dict(raw)
        out = []
        for _, v in d.items():
            e = deepcopy(RETURN_TYPE)
            e[ENTITY] = v.get('country', 'Unknown')
            out.append(e)
        return out or [deepcopy(RETURN_TYPE)]
    return _fn

def get_categories(args):
    row = args['data']
    sa = _safe_get(row, 'subject_areas', None)
    if isinstance(sa, str):
        out = []
        for subj in sa.split(';'):
            if subj.strip():
                e = deepcopy(RETURN_TYPE)
                e[ENTITY] = subj.strip()
                out.append(e)
        return out
    return [deepcopy(RETURN_TYPE)]

def split_string(args, split_with=';'):
    return args['data'].split(split_with)

def get_acronyms(args):
    """Tries columns: 'acronym_attribution', 'acronyms', 'acronym'."""
    row = args.get('data', None)
    if row is None:
        return []
    for col in ('acronym_attribution', 'acronyms', 'acronym'):
        v = _safe_get(row, col, None)
        if v is not None and str(v).strip() and str(v).lower() != 'nan':
            text = str(v).replace(';', ',')
            return list_split_no_attrs(text, split_with=',')
    return []

# ======================================================================================
# Topics & NER maps
# ======================================================================================

def default_topic_triplet_map():
    return {
        'ENTITIES': [
            {ET: TOPIC_TYPE, MAKE_ID_UNIQUE: True, FROM_COL: 'Graph_Name',
             ATTR_COL: [
                {FROM_COL: 'label',       ATTR_NAME: 'label'},
                {FROM_COL: 'Graph_Name',  ATTR_NAME: 'Graph_Name'},
             ]},
            {ET: KEYWORD_TYPE, MAKE_ID_UNIQUE: True},
        ],
        'RELATIONS': [
            {HT: TOPIC_TYPE, R: 'child_of', TT: TOPIC_TYPE, EXTRACT_T: get_parent_topic},
            {HT: TOPIC_TYPE, R: 'mentions', TT: KEYWORD_TYPE, EXTRACT_T: get_topic_keywords},
        ]
    }

def get_parent_topic(args):
    data = args['data']
    graph_name = _get(data, 'Graph_Name') or _get(data, 'graph_name')
    if not isinstance(graph_name, str) or not graph_name.strip():
        return []
    gn = graph_name.strip()
    parent = gn.rsplit('_', 1)[0] if '_' in gn else None
    if not parent:
        return []
    e = deepcopy(RETURN_TYPE)
    e[ENTITY] = parent
    e[ATTRIBUTES] = {"entity_type": "Topic"}
    return [e]

def get_topic_keywords(args):
    data = args['data']
    raw = _get(data, 'keywords') or _get(data, 'words') or _get(data, 'keyword_list')

    items: List[str]
    if raw is None:
        items = []
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        s = str(raw).strip()
        if not s:
            items = []
        else:
            try:
                maybe = ast.literal_eval(s)
                if isinstance(maybe, (list, tuple, set)):
                    items = list(maybe)
                else:
                    items = [maybe]
            except Exception:
                items = re.split(r'[,\|;]\s*', s)

    seen = set()
    keywords = []
    for w in items:
        w = str(w).strip()
        if w and w not in seen:
            seen.add(w)
            keywords.append(w)

    out = []
    for w in keywords:
        e = deepcopy(RETURN_TYPE)
        e[ENTITY] = w
        e[ATTRIBUTES] = {"entity_type": "Keyword"}
        out.append(e)
    return out

# --------------------------- NER
def _parse_ner_cell(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

def make_ner_extractor(label: str,
                       preferred_cols=None,
                       min_len: int = 1,
                       dedupe: bool = True):
    """
    Returns an extractor function bound to `label`.
    Termite will call this with a single dict: {'data': <row>, ...}
    """
    preferred_cols = list(preferred_cols or [])

    def _extract(args, _label=label):
        row = args.get("data")
        if row is None:
            return []

        # discover available keys on the row
        try:
            keys = list(getattr(row, "index", [])) or list(getattr(row, "keys", lambda: [])())
        except Exception:
            keys = []

        # columns to scan: preferred first, then any *_ents_by_label or ner_by_label
        scan_cols = []
        for c in preferred_cols:
            if c in keys:
                scan_cols.append(c)
        for k in keys:
            ks = str(k)
            if ks == "ner_by_label" or ks.endswith("_ents_by_label") or "ents_by_label" in ks:
                if k not in scan_cols:
                    scan_cols.append(k)
        if not scan_cols:
            return []

        seen = set()
        out = []
        for c in scan_cols:
            d = _parse_ner_cell(_safe_get(row, c, None))
            if not isinstance(d, dict):
                continue
            items = d.get(_label, []) or []
            for text in items:
                t = ("" if text is None else str(text)).strip()
                if len(t) < min_len:
                    continue
                if dedupe and t in seen:
                    continue
                seen.add(t)
                ent = deepcopy(RETURN_TYPE)
                ent[ENTITY] = t
                ent[ATTRIBUTES] = [("label", _label), ("source_col", str(c))]
                out.append(ent)
        return out

    return _extract

def default_ner_triplet_map(
    labels=NER_LABELS,
    relation_name="mentions",
    # prefer reading this column first if present; we still auto-detect *_ents_by_label
    ner_col="ner_by_label",
    # choose a reliable document ID column
    document_id_col="doi",
):
    m = {"ENTITIES": [], "RELATIONS": []}

    # HEAD: Document entity present in this pass
    m["ENTITIES"].append({
        ET: DOCUMENT_TYPE,
        FROM_COL: document_id_col,
        MAKE_ID_UNIQUE: True,
    })

    # TAIL: One ET per label
    for lab in labels:
        m["ENTITIES"].append({ET: f"NER_{lab}", MAKE_ID_UNIQUE: True})

    # RELATIONS: bind a label-specific extractor via closure
    for lab in labels:
        m["RELATIONS"].append(
            {
                HT: DOCUMENT_TYPE,
                R: relation_name,
                TT: f"NER_{lab}",
                EXTRACT_T: make_ner_extractor(lab, preferred_cols=[ner_col], min_len=1, dedupe=True),
                PAIRING: HEAD_TO_MANY,
            }
        )
    return m

# ======================================================================================
# Block
#   Runs THREE passes in order: data -> topics -> ner
# ======================================================================================

DEFAULT_CALL_SETTINGS: Dict[str, Any] = {
    # Source CSVs (None → try bundle keys)
    "raw_csv_path": None,          # data/docs
    "topic_csv_path": None,        # topics/labels (NEW) or fallback to data CSV
    # Output files
    "triplets_filename": "triplets.csv",             # legacy (data)
    "data_triplets_filename": None,                  # falls back to triplets_filename
    "topic_triplets_filename": "topic_triplets.csv",
    "ner_triplets_filename": "ner_triplets.csv",
    # Maps
    "column_triplet_map": None,                      # legacy data map
    "column_triplet_map_data": None,                 # falls back to column_triplet_map or default (dynamic below)
    "column_triplet_map_topics": None,               # falls back to default_topic_triplet_map()
    "column_triplet_map_ner": None,                  # falls back to default_ner_triplet_map()
    # NER config
    "ner_labels": NER_LABELS,
    "ner_relation_name": "mentions",
    "ner_col": "ner_by_label",
    # Neo4j creds
    "neo4j_uri": os.getenv("NEO4J_URI", "neo4j://localhost:7666"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_pass": os.getenv("NEO4J_PASS", "local_password"),
    # Optional Termite token
    "token": None,
}

class TermiteNeo4jBlock(AnimalBlock):
    """
    Wrapper that:
      1) builds *data* triplets and pushes to Neo4j
      2) builds *topic* triplets and pushes to Neo4j
      3) builds *NER* triplets and pushes to Neo4j
    """

    CANONICAL_NEEDS: Tuple[str, ...] = ("df", "leaf_labels_csv")

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("data_triplets_csv", "topic_triplets_csv", "ner_triplets_csv"),
        tag: str = "TermiteNeo4j",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        merged_call_settings = {**DEFAULT_CALL_SETTINGS, **(call_settings or {})}
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=init_settings or {},
            call_settings=merged_call_settings,
            verbose=verbose,
            checkpoint=False,
            **kw,
        )

    def _prefer_bundle(self, bundle: DataBundle, *keys: str) -> Optional[str]:
        for k in keys:
            try:
                v = bundle.get(k)
            except Exception:
                v = None
            if v:
                return v
        return None

    def run(self, bundle: DataBundle) -> None:
        root_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]).expanduser().resolve()
        out_dir = (root_dir / self.tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Resolve CSV inputs ----------
        data_input = bundle[self.needs[0]]  # typically a DataFrame (spaceyNER output)
        out_dir = (Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        def _ensure_csv_path(obj, fallback_name):
            from pathlib import Path
            import pandas as pd
            if isinstance(obj, (str, Path)):
                return Path(str(obj)).expanduser().resolve()
            # If a DataFrame was passed in the bundle, persist it for Termite
            if hasattr(obj, "to_csv"):
                p = out_dir / fallback_name
                obj.to_csv(p, index=False, encoding="utf-8-sig")
                return p
            # Last resort: try to interpret as string path
            return Path(str(obj)).expanduser().resolve()

        # Use the enriched DF for DATA and NER passes
        data_csv_path = _ensure_csv_path(data_input, "termite_input_data.csv")

        # Topics can be a separate CSV (leaf labels); fall back to data if not provided
        topic_input = (
            self.call_settings.get("topic_csv_path")
            or bundle.get(self.needs[1])  # typically "leaf_labels_csv"
            or data_input
        )
        topic_csv_path = _ensure_csv_path(topic_input, "termite_input_topics.csv")

        # ---------- Inspect columns / ensure a usable document id ----------
        try:
            cols = set(pd.read_csv(data_csv_path, nrows=0).columns)
        except Exception:
            cols = set()

        def _first_present(cands: Sequence[str]) -> Optional[str]:
            for c in cands:
                if c in cols:
                    return c
            return None

        # Pick a document ID column; create one if needed
        doc_id_col = _first_present(["doi", "s2id", "eid"])
        if doc_id_col is None:
            df_all = pd.read_csv(data_csv_path)
            if "doi" in df_all.columns and df_all["doi"].notna().any():
                doc_id_col = "doi"
            elif "s2id" in df_all.columns and df_all["s2id"].notna().any():
                doc_id_col = "s2id"
            elif "eid" in df_all.columns and df_all["eid"].notna().any():
                doc_id_col = "eid"
            else:
                df_all["doc_id"] = [f"row-{i}" for i in range(len(df_all))]
                doc_id_col = "doc_id"
            # make sure year exists (some upstreams omit it)
            if "year" not in df_all.columns:
                df_all["year"] = 0
            df_all.to_csv(data_csv_path, index=False, encoding="utf-8-sig")
            cols = set(df_all.columns)

        # Build attribute list only from columns that actually exist
        attr_cols = []
        if "title" in cols:
            attr_cols.append({FROM_COL: "title", ATTR_NAME: "Title"})
        for c, name in (("eid", "EID"), ("s2id", "S2ID"), ("doi", "DOI")):
            if c in cols:
                attr_cols.append({FROM_COL: c, ATTR_NAME: name})

        # Detect author id/name columns
        author_ids_col = _first_present(["slic_author_ids", "s2_author_ids", "author_ids"]) or "author_ids"
        authors_col    = _first_present(["slic_authors", "s2_authors", "authors"]) or "authors"

        # Detect affiliations column
        aff_col = _first_present(["slic_affiliations", "affiliations"]) or "affiliations"

        # If 'year' is missing entirely, add a dummy year so Year nodes can be created
        if "year" not in cols:
            df_all = pd.read_csv(data_csv_path)
            df_all["year"] = 0
            df_all.to_csv(data_csv_path, index=False, encoding="utf-8-sig")
            cols = set(df_all.columns)

        # ---------- Resolve outputs ----------
        data_triplets_filename = (
            self.call_settings.get("data_triplets_filename")
            or self.call_settings.get("triplets_filename", "triplets.csv")
        )
        topic_triplets_filename = self.call_settings.get("topic_triplets_filename", "topic_triplets.csv")
        ner_triplets_filename   = self.call_settings.get("ner_triplets_filename", "ner_triplets.csv")

        data_triplets_path  = out_dir / data_triplets_filename
        topic_triplets_path = out_dir / topic_triplets_filename
        ner_triplets_path   = out_dir / ner_triplets_filename

        # ---------- Build (or take) triplet maps ----------
        # DATA map: if caller didn’t provide one, build a column‑aware map
        provided_data_map = self.call_settings.get("column_triplet_map_data") or self.call_settings.get("column_triplet_map")

        if provided_data_map:
            data_triplet_map = provided_data_map
        else:
            data_triplet_map = {
                'ENTITIES': [
                    {ET: TOPIC_TYPE, MAKE_ID_UNIQUE: True, FROM_COL: 'Graph_Name'},
                    {ET: DOCUMENT_TYPE, FROM_COL: doc_id_col, ATTR_COL: attr_cols, MAKE_ID_UNIQUE: True},
                    {ET: AFFILIATION_IDENTIFIER_TYPE, MAKE_ID_UNIQUE: True},
                    {ET: COUNTRY_TYPE,                MAKE_ID_UNIQUE: True},
                    {ET: CATEGORY, MAKE_ID_UNIQUE: True},
                    {ET: ACRONYM,  MAKE_ID_UNIQUE: True},
                    {ET: YEAR_TYPE, FROM_COL: 'year', ATTR_COL: None, ATTR_FUNC: None, MAKE_ID_UNIQUE: True},
                    {
                        ET: AUTHOR_ID_TYPE,
                        FROM_COL: author_ids_col,
                        ATTR_COL: [{FROM_COL: authors_col, ATTR_NAME: 'Author_Name', RETREIVAL: split_string, ARGS: None}],
                        ATTR_FUNC: split_string, ARGS: None, MAKE_ID_UNIQUE: True
                    },
                    {ET: PUBLISHER, FROM_COL: 'publication_name', MAKE_ID_UNIQUE: True},
                ],
                'RELATIONS': [
                    {HT: DOCUMENT_TYPE, R: 'part_of_topic',           TT: TOPIC_TYPE},
                    {HT: DOCUMENT_TYPE, R: DOCUMENT_YEAR_RELATION,    TT: YEAR_TYPE},
                    {
                        HT: AUTHOR_ID_TYPE, R: AUTHOR_DOCUMENT_RELATION, TT: DOCUMENT_TYPE,
                        EXTRACT_H: make_get_authors_ID_from(author_ids_col, authors_col)
                    },
                    {
                        HT: DOCUMENT_TYPE, R: DOCUMENT_AFFILITATION_RELATION, TT: AFFILIATION_IDENTIFIER_TYPE,
                        EXTRACT_T: make_get_affiliations_from(aff_col)
                    },
                    {
                        HT: AFFILIATION_IDENTIFIER_TYPE, R: AFFILIATION_COUNTRY_RELATION, TT: COUNTRY_TYPE,
                        EXTRACT_H: make_get_affiliations_from(aff_col),
                        EXTRACT_T: make_get_countries_from(aff_col),
                        PAIRING: INDEX_PAIRING
                    },
                    {HT: DOCUMENT_TYPE, R: DOCUMENT_PUBLISHER_RELATION, TT: PUBLISHER},
                    {HT: DOCUMENT_TYPE, R: DOCUMENT_CATEGORY_RELATION,  TT: CATEGORY,  EXTRACT_T: get_categories, PAIRING: HEAD_TO_MANY},
                    {HT: DOCUMENT_TYPE, R: DOCUMENT_ACRONYM_RELATION,   TT: ACRONYM,   EXTRACT_T: get_acronyms},
                ],
            }

        topic_triplet_map = self.call_settings.get("column_triplet_map_topics") or default_topic_triplet_map()

        ner_triplet_map = (
            self.call_settings.get("column_triplet_map_ner")
            or default_ner_triplet_map(
                labels=self.call_settings.get("ner_labels", NER_LABELS),
                relation_name=self.call_settings.get("ner_relation_name", "mentions"),
                ner_col=self.call_settings.get("ner_col", "ner_by_label"),
                document_id_col=doc_id_col,
            )
        )

        # ---------- Neo4j / Termite ----------
        neo4j_uri  = self.call_settings["neo4j_uri"]
        neo4j_user = self.call_settings["neo4j_user"]
        neo4j_pass = self.call_settings["neo4j_pass"]
        token = self.call_settings.get("token", None)

        termite = Termite(
            kg_credentials=(neo4j_uri, (neo4j_user, neo4j_pass)),
            vector_uri=None,
            db_nme="default",
            token=token,
            verbose=self.verbose,
        )

        # Create uniqueness constraints for all three schemas up front
        termite.make_unique_constrains(data_triplet_map)
        termite.make_unique_constrains(topic_triplet_map)
        termite.make_unique_constrains(ner_triplet_map)

        # ---------- PASS 1: DATA ----------
        termite.from_csv_to_triplets(str(data_csv_path), str(data_triplets_path), data_triplet_map)
        termite.update_database_multithreaded(str(data_triplets_path))

        # ---------- PASS 2: TOPICS ----------
        termite.from_csv_to_triplets(str(topic_csv_path), str(topic_triplets_path), topic_triplet_map)
        termite.update_database_multithreaded(str(topic_triplets_path))

        # ---------- PASS 3: NER ----------
        termite.from_csv_to_triplets(str(data_csv_path), str(ner_triplets_path), ner_triplet_map)
        termite.update_database_multithreaded(str(ner_triplets_path))

        # ---------- Register outputs ----------
        self.register_checkpoint("data_triplets_csv", data_triplets_path)
        self.register_checkpoint("topic_triplets_csv", topic_triplets_path)
        self.register_checkpoint("ner_triplets_csv", ner_triplets_path)

        bundle[f"{self.tag}.data_triplets_csv"] = data_triplets_path
        bundle[f"{self.tag}.topic_triplets_csv"] = topic_triplets_path
        bundle[f"{self.tag}.ner_triplets_csv"]   = ner_triplets_path

        if self.verbose:
            print(f"[{self.tag}] Data triplets  @ {data_triplets_path}")
            print(f"[{self.tag}] Topic triplets @ {topic_triplets_path}")
            print(f"[{self.tag}] NER triplets   @ {ner_triplets_path}")
