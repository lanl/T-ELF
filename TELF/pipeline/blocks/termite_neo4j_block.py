# TELF/pipeline/blocks/termite_neo4j_block.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional, List, Callable
import pandas as pd
from copy import deepcopy
import ast
import json
import re
import yaml

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

# --------------------------- NER default labels
NER_LABELS = [
    "ORG", "PERSON", "GPE", "NORP", "FAC", "LOC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
]

# ======================================================================================
# General helpers
# ======================================================================================
def _map_has_entities(m: Optional[dict]) -> bool:
    return bool(m and isinstance(m, dict) and isinstance(m.get("ENTITIES"), list) and len(m["ENTITIES"]) > 0)

def _run_pass_if_nonempty(
    *,
    termite,
    csv_path: Path,
    triplets_path: Path,
    triplet_map: dict,
    pass_name: str,
    verbose: bool = True
) -> bool:
    """
    Returns True if the pass ran; False if skipped due to empty ENTITIES.
    """
    if not _map_has_entities(triplet_map):
        if verbose:
            print(f"[TermiteNeo4j] Skipping '{pass_name}' pass: no ENTITIES defined.")
        return False

    # Create constraints and run
    termite.make_unique_constrains(triplet_map)
    termite.from_csv_to_triplets(str(csv_path), str(triplets_path), triplet_map)
    termite.update_database_multithreaded(str(triplets_path))
    if verbose:
        print(f"[TermiteNeo4j] Finished '{pass_name}' pass -> {triplets_path}")
    return True


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
            v = v.strip()
            if not v:
                continue
            e = deepcopy(RETURN_TYPE)
            e[ENTITY] = v
            out.append(e)
        return out or [deepcopy(RETURN_TYPE)]
    return [deepcopy(RETURN_TYPE)]

def split_string(args, split_with=';'):
    s = args.get('data')
    if isinstance(s, str):
        return [t for t in (x.strip() for x in s.split(split_with)) if t]
    return []

# ======================================================================================
# Domain extractors (authors, affiliations, categories, acronyms)
# ======================================================================================

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

def _aff_to_dict(cell) -> Dict[str, Dict[str, Any]]:
    """Accept dict/list/JSON/string; return dict keyed by affiliation id; ensure name/authors/country keys."""
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
            subj = subj.strip()
            if subj:
                e = deepcopy(RETURN_TYPE)
                e[ENTITY] = subj
                out.append(e)
        return out or [deepcopy(RETURN_TYPE)]
    return [deepcopy(RETURN_TYPE)]

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
# Topics helpers
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

# ======================================================================================
# NER helpers
# ======================================================================================

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

def make_ner_extractor(label: str, preferred_cols=None, min_len: int = 1, dedupe: bool = True):
    preferred_cols = list(preferred_cols or [])
    def _extract(args, _label=label):
        row = args.get("data")
        if row is None:
            return []
        try:
            keys = list(getattr(row, "index", [])) or list(getattr(row, "keys", lambda: [])())
        except Exception:
            keys = []
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

def default_ner_triplet_map(labels=NER_LABELS, relation_name="mentions", ner_col="ner_by_label", document_id_col="doi"):
    m = {"ENTITIES": [], "RELATIONS": []}
    m["ENTITIES"].append({ET: DOCUMENT_TYPE, FROM_COL: document_id_col, MAKE_ID_UNIQUE: True})
    for lab in labels:
        m["ENTITIES"].append({ET: f"NER_{lab}", MAKE_ID_UNIQUE: True})
    for lab in labels:
        m["RELATIONS"].append({
            HT: DOCUMENT_TYPE, R: relation_name, TT: f"NER_{lab}",
            EXTRACT_T: make_ner_extractor(lab, preferred_cols=[ner_col], min_len=1, dedupe=True),
            PAIRING: HEAD_TO_MANY,
        })
    return m

# ======================================================================================
# YAML (simple) : head/relation/tail + function
# ======================================================================================

def _interpolate_env(value, env):
    """Resolve ${path.to.value} inside strings by walking the loaded YAML dict."""
    if isinstance(value, str):
        for path in re.findall(r"\$\{([^}]+)\}", value):
            cur = env
            for p in path.split("."):
                cur = cur[p]
            value = value.replace("${"+path+"}", str(cur))
        return value
    if isinstance(value, list):
        return [_interpolate_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate_env(v, env) for k, v in value.items()}
    return value

def load_settings_yml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _interpolate_env(raw, raw)

def split_factory(sep):
    def _split(args, split_with=sep):
        return split_string({'data': _safe_get(args, 'data')}, split_with=split_with)
    return _split

def resolve_unpacker(name: str, args_cfg: Optional[dict]) -> Callable:
    """
    Supported built-ins:
      - split (args: {sep})
      - get_authors_ID (args: {ids_col, names_col})
      - get_affiliations / get_countries (args: {aff_col})
      - get_topic_keywords / get_parent_topic / get_categories / get_acronyms
    Optional dynamic loader (commented below) can support "python:module.func".
    """
    args_cfg = args_cfg or {}
    if name == "split":
        return split_factory(args_cfg.get("sep", ";"))
    if name == "get_authors_ID":
        return make_get_authors_ID_from(args_cfg.get("ids_col", "author_ids"), args_cfg.get("names_col", "authors"))
    if name == "get_affiliations":
        return make_get_affiliations_from(args_cfg.get("aff_col", "affiliations"))
    if name == "get_countries":
        return make_get_countries_from(args_cfg.get("aff_col", "affiliations"))
    if name == "get_topic_keywords":
        return get_topic_keywords
    if name == "get_parent_topic":
        return get_parent_topic
    if name == "get_categories":
        return get_categories
    if name == "get_acronyms":
        return get_acronyms

    # # Optional: dynamic dotted-path loader
    # if name.startswith("python:"):
    #     mod_path = name.split("python:", 1)[1]
    #     module, func = mod_path.rsplit(".", 1)
    #     m = __import__(module, fromlist=[func])
    #     f = getattr(m, func)
    #     def _wrapped(args, f=f, a=args_cfg or {}):
    #         return f(args, **a) if a else f(args)
    #     return _wrapped

    raise ValueError(f"Unknown function '{name}'")

def _coalesce_unpacker(spec: Optional[dict] | str) -> Optional[Callable]:
    """Accept string ('get_categories') or mapping {function: 'split', args: {...}}."""
    if not spec:
        return None
    if isinstance(spec, str):
        return resolve_unpacker(spec, {})
    fn_name = spec.get("function") or spec.get("fn")
    if not fn_name:
        return None
    return resolve_unpacker(fn_name, spec.get("args", {}))

def entity_from_simple_yaml(e: dict) -> dict:
    """
    Simple keys:
      - type (required) -> ET
      - from (optional) -> FROM_COL
      - unique: bool -> MAKE_ID_UNIQUE
      - attrs: list of {from, as, function?}
    """
    et = e.get("type") or e.get("et")
    if not et:
        raise ValueError("Entity requires 'type'")

    out = {ET: et, MAKE_ID_UNIQUE: bool(e.get("unique", e.get("make_id_unique", True)))}

    src = e.get("from", e.get("from_col", None))
    if src is not None:
        out[FROM_COL] = src

    attrs = e.get("attrs", [])
    if attrs:
        cols = []
        for a in attrs:
            a_from = a.get("from") or a.get("from_col")
            a_as   = a.get("as")   or a.get("attr_name")
            if not a_from or not a_as:
                continue
            col = {FROM_COL: a_from, ATTR_NAME: a_as}
            if a.get("function") or a.get("fn"):
                func_spec = a.get("function") or a.get("fn")
                func = _coalesce_unpacker(func_spec)
                col[RETREIVAL] = func
                col[ARGS] = (func_spec.get("args") if isinstance(func_spec, dict) else None)
            cols.append(col)
        if cols:
            out[ATTR_COL] = cols

    if e.get("function") or e.get("fn"):
        out[ATTR_FUNC] = _coalesce_unpacker(e.get("function") or e.get("fn"))
    if e.get("args"):
        out[ARGS] = e["args"]

    return out

def relation_from_simple_yaml(r: dict) -> dict:
    """
    Simple keys:
      - head, relation, tail
      - head_extract / tail_extract (string or {function, args})
      - pairing (HEAD_TO_MANY, INDEX_PAIRING, etc.)
    """
    head = r.get("head") or r.get("ht")
    rel  = r.get("relation") or r.get("r")
    tail = r.get("tail") or r.get("tt")
    if not (head and rel and tail):
        raise ValueError("Relation requires 'head', 'relation', and 'tail'")

    out = {HT: head, R: rel, TT: tail}
    if r.get("pairing"):
        out[PAIRING] = r["pairing"]

    he = r.get("head_extract") or r.get("extract_h")
    te = r.get("tail_extract") or r.get("extract_t")
    if he:
        out[EXTRACT_H] = _coalesce_unpacker(he)
    if te:
        out[EXTRACT_T] = _coalesce_unpacker(te)
    return out

def build_triplet_map_from_simple_yaml(section: dict) -> dict:
    """Convert YAML section (data/topics) to Termite triplet-map dict (simple schema)."""
    m = {"ENTITIES": [], "RELATIONS": []}
    for e in section.get("entities", []):
        m["ENTITIES"].append(entity_from_simple_yaml(e))
    for r in section.get("relations", []):
        m["RELATIONS"].append(relation_from_simple_yaml(r))
    return m

def build_ner_triplet_map_from_yaml(ner_cfg: dict, *, doc_col_fallback="doi", ner_col_fallback="ner_by_label") -> dict:
    """NER map from YAML; falls back to detected columns if keys omitted."""
    doc_col = ner_cfg.get("document_id_col", doc_col_fallback)
    ner_col = ner_cfg.get("ner_col", ner_col_fallback)
    relation_name = ner_cfg.get("relation_name", "mentions")
    labels = ner_cfg.get("labels", [])

    m = {"ENTITIES": [], "RELATIONS": []}
    m["ENTITIES"].append({ET: DOCUMENT_TYPE, FROM_COL: doc_col, MAKE_ID_UNIQUE: True})
    for lab in labels:
        m["ENTITIES"].append({ET: f"NER_{lab}", MAKE_ID_UNIQUE: True})
    for lab in labels:
        extractor = make_ner_extractor(lab, preferred_cols=[ner_col], min_len=1, dedupe=True)
        m["RELATIONS"].append({
            HT: DOCUMENT_TYPE, R: relation_name, TT: f"NER_{lab}",
            EXTRACT_T: extractor, PAIRING: HEAD_TO_MANY,
        })
    return m

def merge_maps(base: dict, extra: dict) -> dict:
    """Concatenate ENTITIES/RELATIONS lists (Neo4j constraints handle uniqueness)."""
    out = {"ENTITIES": list(base.get("ENTITIES", [])), "RELATIONS": list(base.get("RELATIONS", []))}
    out["ENTITIES"].extend(extra.get("ENTITIES", []))
    out["RELATIONS"].extend(extra.get("RELATIONS", []))
    return out




def _prune_empty_node_constraints(
    *,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    exclude_labels: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> None:
    """
    Drops NODE constraints whose target label set currently matches 0 nodes.
    Requires Neo4j 4.4+ (SHOW CONSTRAINTS). Uses constraint names for DROP.
    """
    try:
        from neo4j import GraphDatabase
    except Exception as e:
        if verbose:
            print(f"[TermiteNeo4j] Prune skipped: neo4j driver not installed ({e})")
        return

    exclude = set(exclude_labels or [])

    def _bt(s: str) -> str:
        # backtick-escape for Cypher identifiers
        return s.replace("`", "``")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as sess:
            rows = sess.run(
                "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
                "RETURN name, type, entityType, labelsOrTypes, properties"
            ).data()

            drop_names: list[str] = []
            for r in rows:
                if (r.get("entityType") != "NODE") or not r.get("labelsOrTypes"):
                    continue

                labels: list[str] = list(r["labelsOrTypes"])
                if any(lab in exclude for lab in labels):
                    continue

                # Build a label pattern that requires *all* labels (LabelA:LabelB)
                label_pattern = ":" + ":".join(f"`{_bt(l)}`" for l in labels)
                count_rec = sess.run(f"MATCH (n{label_pattern}) RETURN count(n) AS c").single()
                count_val = int(count_rec["c"]) if count_rec else 0

                if count_val == 0:
                    name = r.get("name")
                    if name:
                        drop_names.append(name)

            for nm in drop_names:
                sess.run(f"DROP CONSTRAINT `{_bt(nm)}` IF EXISTS")
                if verbose:
                    print(f"[TermiteNeo4j] Dropped empty-node constraint `{nm}`")
    finally:
        driver.close()

# ======================================================================================
# Mode parsing
# ======================================================================================

def _parse_yaml_mode(cfg: dict) -> int:
    """
    Returns 1 (only), 2 (merge, default), 3 (ignore).
    Accepts numbers or strings: 1/'only', 2/'merge', 3/'ignore'.
    """
    v = cfg.get("mode", 2)  # default merge
    s = str(v).strip().lower()
    if s in ("1", "only", "yaml_only", "strict"):
        return 1
    if s in ("3", "ignore", "off", "none", "disabled"):
        return 3
    return 2  # merge

# ======================================================================================
# Block
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
    "column_triplet_map": None,
    "column_triplet_map_data": None,
    "column_triplet_map_topics": None,
    "column_triplet_map_ner": None,
    # NER config
    "ner_labels": NER_LABELS,
    "ner_relation_name": "mentions",
    "ner_col": "ner_by_label",
    # YAML (optional)
    "settings_yaml_path": None,
    # Neo4j creds
    "neo4j_uri": os.getenv("NEO4J_URI", "neo4j://localhost:7666"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_pass": os.getenv("NEO4J_PASS", "local_password"),
    # Optional Termite token
    "token": None,

    # --- NEW: constraint pruning ---
    "neo4j_db": os.getenv("NEO4J_DB", "neo4j"),     # DB name used by SHOW/DROP
    "prune_empty_constraints": False,                # opt-in
    "prune_when": "after",                           # "before" | "after" | "both"
    "prune_labels_exclude": [],                      # optional: don't drop for these labels
}

class TermiteNeo4jBlock(AnimalBlock):
    """
    Supports YAML `mode`:
      1=only       -> use ONLY what’s defined in YAML (no defaults)
      2=merge      -> defaults + YAML injections (and run `extra_schemas`) [default]
      3=ignore     -> ignore YAML entirely (defaults only; skip extras)

    Builds three base passes (data / topics / NER) and any number of `extra_schemas`.
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
        data_input = bundle[self.needs[0]]  # e.g., a DataFrame (NER-enriched)
        out_dir = (Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        def _ensure_csv_path(obj, fallback_name):
            if isinstance(obj, (str, Path)):
                return Path(str(obj)).expanduser().resolve()
            if hasattr(obj, "to_csv"):
                p = out_dir / fallback_name
                obj.to_csv(p, index=False, encoding="utf-8-sig")
                return p
            return Path(str(obj)).expanduser().resolve()

        data_csv_path = _ensure_csv_path(data_input, "termite_input_data.csv")
        topic_input = (
            self.call_settings.get("topic_csv_path")
            or bundle.get(self.needs[1])  # "leaf_labels_csv"
            or data_input
        )
        topic_csv_path = _ensure_csv_path(topic_input, "termite_input_topics.csv")

        # ---------- Peek & normalize essential columns ----------
        need_full = False
        try:
            cols = set(pd.read_csv(data_csv_path, nrows=0).columns)
            if not {"doi", "s2id", "eid"} & cols or "year" not in cols:
                need_full = True
        except Exception:
            need_full = True

        df_all = None
        if need_full:
            df_all = pd.read_csv(data_csv_path)

        if df_all is not None:
            if not ({"doi","s2id","eid"} & set(df_all.columns)):
                df_all["doc_id"] = [f"row-{i}" for i in range(len(df_all))]
            if "year" not in df_all.columns:
                df_all["year"] = 0
            df_all.to_csv(data_csv_path, index=False, encoding="utf-8-sig")
            cols = set(df_all.columns)

        def _first_present(cands: Sequence[str]) -> Optional[str]:
            for c in cands:
                if c in cols:
                    return c
            return None

        doc_id_col = _first_present(["doi","s2id","eid"]) or ("doc_id" if "doc_id" in cols else "doi")

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

        # ---------- Output paths ----------
        data_triplets_filename = self.call_settings.get("data_triplets_filename") or self.call_settings.get("triplets_filename", "triplets.csv")
        topic_triplets_filename = self.call_settings.get("topic_triplets_filename", "topic_triplets.csv")
        ner_triplets_filename   = self.call_settings.get("ner_triplets_filename", "ner_triplets.csv")

        data_triplets_path  = out_dir / data_triplets_filename
        topic_triplets_path = out_dir / topic_triplets_filename
        ner_triplets_path   = out_dir / ner_triplets_filename

        # ---------- Default triplet maps ----------
        base_data_map = {
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
                {HT: AUTHOR_ID_TYPE, R: AUTHOR_DOCUMENT_RELATION, TT: DOCUMENT_TYPE,
                 EXTRACT_H: make_get_authors_ID_from(author_ids_col, authors_col)},
                {HT: DOCUMENT_TYPE, R: DOCUMENT_AFFILITATION_RELATION, TT: AFFILIATION_IDENTIFIER_TYPE,
                 EXTRACT_T: make_get_affiliations_from(aff_col)},
                {HT: AFFILIATION_IDENTIFIER_TYPE, R: AFFILIATION_COUNTRY_RELATION, TT: COUNTRY_TYPE,
                 EXTRACT_H: make_get_affiliations_from(aff_col),
                 EXTRACT_T: make_get_countries_from(aff_col), PAIRING: INDEX_PAIRING},
                {HT: DOCUMENT_TYPE, R: DOCUMENT_PUBLISHER_RELATION, TT: PUBLISHER},
                {HT: DOCUMENT_TYPE, R: DOCUMENT_CATEGORY_RELATION,  TT: CATEGORY,
                 EXTRACT_T: get_categories, PAIRING: HEAD_TO_MANY},
                {HT: DOCUMENT_TYPE, R: DOCUMENT_ACRONYM_RELATION,   TT: ACRONYM,
                 EXTRACT_T: get_acronyms},
            ],
        }

        base_topics_map = default_topic_triplet_map()
        base_ner_map = default_ner_triplet_map(
            labels=self.call_settings.get("ner_labels", NER_LABELS),
            relation_name=self.call_settings.get("ner_relation_name", "mentions"),
            ner_col=self.call_settings.get("ner_col", "ner_by_label"),
            document_id_col=doc_id_col,
        )

        # ---------- Load YAML (optional) and parse mode ----------
        settings_yaml_path = self.call_settings.get("settings_yaml_path")
        yaml_cfg: dict = {}
        yaml_root: Optional[Path] = None
        yaml_mode: int = 2  # default merge
        if settings_yaml_path:
            settings_yaml_path = str(Path(settings_yaml_path).expanduser().resolve())
            yaml_root = Path(settings_yaml_path).parent
            yaml_cfg = load_settings_yml(settings_yaml_path) or {}
            yaml_mode = _parse_yaml_mode(yaml_cfg)

            # optional creds override from YAML (applies for modes 1 and 2)
            if yaml_mode in (1, 2) and "neo4j" in yaml_cfg:
                self.call_settings["neo4j_uri"]  = yaml_cfg["neo4j"].get("uri", self.call_settings["neo4j_uri"])
                self.call_settings["neo4j_user"] = yaml_cfg["neo4j"].get("user", self.call_settings["neo4j_user"])
                self.call_settings["neo4j_pass"] = yaml_cfg["neo4j"].get("pass", self.call_settings["neo4j_pass"])
        else:
            yaml_mode = 3  # treat as ignore if no path provided

        # ---------- Build maps per mode ----------
        if yaml_mode == 3:
            # IGNORE: behave as if no YAML provided (defaults only; skip extras)
            provided_data_map = self.call_settings.get("column_triplet_map_data") or self.call_settings.get("column_triplet_map")
            data_triplet_map = provided_data_map or base_data_map
            topic_triplet_map = self.call_settings.get("column_triplet_map_topics") or base_topics_map
            ner_triplet_map = self.call_settings.get("column_triplet_map_ner") or base_ner_map
            extra_schemas_cfg = []

        elif yaml_mode == 1:
            # ONLY YAML: no defaults at all; missing sections -> empty maps
            if yaml_cfg.get("data"):
                data_triplet_map = build_triplet_map_from_simple_yaml(yaml_cfg["data"])
            else:
                data_triplet_map = {"ENTITIES": [], "RELATIONS": []}

            if yaml_cfg.get("topics"):
                topic_triplet_map = build_triplet_map_from_simple_yaml(yaml_cfg["topics"])
            else:
                topic_triplet_map = {"ENTITIES": [], "RELATIONS": []}

            if yaml_cfg.get("ner"):
                ner_triplet_map = build_ner_triplet_map_from_yaml(
                    yaml_cfg["ner"], doc_col_fallback=doc_id_col,
                    ner_col_fallback=self.call_settings.get("ner_col", "ner_by_label"),
                )
            else:
                ner_triplet_map = {"ENTITIES": [], "RELATIONS": []}

            extra_schemas_cfg = yaml_cfg.get("extra_schemas", [])  # run exactly as provided

        else:
            # MERGE: defaults + YAML injections (and run extras)
            if yaml_cfg.get("data"):
                yaml_data = build_triplet_map_from_simple_yaml(yaml_cfg["data"])
                data_triplet_map = merge_maps(base_data_map, yaml_data)
            else:
                provided_data_map = self.call_settings.get("column_triplet_map_data") or self.call_settings.get("column_triplet_map")
                data_triplet_map = provided_data_map or base_data_map

            if yaml_cfg.get("topics"):
                yaml_topics = build_triplet_map_from_simple_yaml(yaml_cfg["topics"])
                topic_triplet_map = merge_maps(base_topics_map, yaml_topics)
            else:
                topic_triplet_map = self.call_settings.get("column_triplet_map_topics") or base_topics_map

            if yaml_cfg.get("ner"):
                yaml_ner = build_ner_triplet_map_from_yaml(
                    yaml_cfg["ner"], doc_col_fallback=doc_id_col,
                    ner_col_fallback=self.call_settings.get("ner_col", "ner_by_label"),
                )
                ner_triplet_map = merge_maps(base_ner_map, yaml_ner)
            else:
                ner_triplet_map = self.call_settings.get("column_triplet_map_ner") or base_ner_map

            extra_schemas_cfg = yaml_cfg.get("extra_schemas", [])

        if yaml_mode in (1, 2):
            # allow both under neo4j: {...} and at top-level for convenience
            neo4j_cfg = yaml_cfg.get("neo4j", {})
            self.call_settings["neo4j_db"] = neo4j_cfg.get("db", self.call_settings.get("neo4j_db", "neo4j"))
            self.call_settings["prune_empty_constraints"] = neo4j_cfg.get(
                "prune_empty_constraints",
                yaml_cfg.get("prune_empty_constraints", self.call_settings["prune_empty_constraints"])
            )
            self.call_settings["prune_when"] = neo4j_cfg.get(
                "prune_when",
                yaml_cfg.get("prune_when", self.call_settings.get("prune_when", "after"))
            )
            self.call_settings["prune_labels_exclude"] = neo4j_cfg.get(
                "prune_labels_exclude",
                yaml_cfg.get("prune_labels_exclude", self.call_settings.get("prune_labels_exclude", []))
            )

        # ---------- Neo4j / Termite ----------
        neo4j_uri  = self.call_settings["neo4j_uri"]
        neo4j_user = self.call_settings["neo4j_user"]
        neo4j_pass = self.call_settings["neo4j_pass"]
        token = self.call_settings.get("token", None)

        if self.call_settings.get("prune_empty_constraints") and self.call_settings.get("prune_when") in ("before", "both"):
            _prune_empty_node_constraints(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_pass,
                database=self.call_settings.get("neo4j_db", "neo4j"),
                exclude_labels=self.call_settings.get("prune_labels_exclude", []),
                verbose=self.verbose,
            )

        termite = Termite(
            kg_credentials=(neo4j_uri, (neo4j_user, neo4j_pass)),
            vector_uri=None,
            db_nme="default",
            token=token,
            verbose=self.verbose,
        )

        # Create uniqueness constraints for base maps up front
        # ---------- Base passes (guarded) ----------
        ran_data = _run_pass_if_nonempty(
            termite=termite,
            csv_path=Path(data_csv_path),
            triplets_path=data_triplets_path,
            triplet_map=data_triplet_map,
            pass_name="data",
            verbose=self.verbose,
        )

        ran_topics = _run_pass_if_nonempty(
            termite=termite,
            csv_path=Path(topic_csv_path),
            triplets_path=topic_triplets_path,
            triplet_map=topic_triplet_map,
            pass_name="topics",
            verbose=self.verbose,
        )

        ran_ner = _run_pass_if_nonempty(
            termite=termite,
            csv_path=Path(data_csv_path),
            triplets_path=ner_triplets_path,
            triplet_map=ner_triplet_map,
            pass_name="ner",
            verbose=self.verbose,
        )


        # ---------- OPTIONAL: any number of extra schema passes (modes 1 & 2 only) ----------
        extra_passes_info: List[Tuple[str, Path]] = []
        if extra_schemas_cfg and isinstance(extra_schemas_cfg, list):
            for i, sch in enumerate(extra_schemas_cfg):
                if not isinstance(sch, dict):
                    continue
                # Name & CSV
                name = str(sch.get("name", f"extra_{i}")).strip() or f"extra_{i}"
                csv_arg = sch.get("csv", None)
                if csv_arg:
                    p = Path(csv_arg)
                    csv_path = (yaml_root / p).resolve() if not p.is_absolute() and yaml_root else p.resolve()
                else:
                    csv_path = data_csv_path  # default to main data CSV

                # Build map from the same simple schema keys
                has_map_bits = bool(sch.get("entities") or sch.get("relations"))
                if has_map_bits:
                    schema_map = build_triplet_map_from_simple_yaml(sch)
                else:
                    # allow 'map:' nested
                    if isinstance(sch.get("map"), dict):
                        schema_map = build_triplet_map_from_simple_yaml(sch["map"])
                    else:
                        continue  # nothing to do

                # Output file for this extra pass
                extra_triplets_filename = sch.get("triplets_filename", f"{name}_triplets.csv")
                extra_triplets_path = out_dir / extra_triplets_filename

                # Execute pass (guarded)
                if _run_pass_if_nonempty(
                    termite=termite,
                    csv_path=csv_path,
                    triplets_path=extra_triplets_path,
                    triplet_map=schema_map,
                    pass_name=f"extra:{name}",
                    verbose=self.verbose,
                ):
                    bundle[f"{self.tag}.{name}_triplets_csv"] = extra_triplets_path
                    extra_passes_info.append((name, extra_triplets_path))
                else:
                    if self.verbose:
                        print(f"[{self.tag}] Extra schema '{name}' skipped (no ENTITIES).")


                # Register & log
                bundle[f"{self.tag}.{name}_triplets_csv"] = extra_triplets_path
                extra_passes_info.append((name, extra_triplets_path))


        if self.call_settings.get("prune_empty_constraints") and self.call_settings.get("prune_when") in ("after", "both"):
            _prune_empty_node_constraints(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_pass,
                database=self.call_settings.get("neo4j_db", "neo4j"),
                exclude_labels=self.call_settings.get("prune_labels_exclude", []),
                verbose=self.verbose,
            )

        # ---------- Register base outputs ----------
        self.register_checkpoint("data_triplets_csv", data_triplets_path)
        self.register_checkpoint("topic_triplets_csv", topic_triplets_path)
        self.register_checkpoint("ner_triplets_csv", ner_triplets_path)

        bundle[f"{self.tag}.data_triplets_csv"]  = data_triplets_path
        bundle[f"{self.tag}.topic_triplets_csv"] = topic_triplets_path
        bundle[f"{self.tag}.ner_triplets_csv"]   = ner_triplets_path

        if self.verbose:
            print(f"[{self.tag}] Data triplets  @ {data_triplets_path}")
            print(f"[{self.tag}] Topic triplets @ {topic_triplets_path}")
            print(f"[{self.tag}] NER triplets   @ {ner_triplets_path}")
            if extra_passes_info:
                for nm, pth in extra_passes_info:
                    print(f"[{self.tag}] Extra schema '{nm}' triplets @ {pth}")
