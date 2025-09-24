# TELF/pipeline/blocks/termite_neo4j_block.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional
import pandas as pd
from copy import deepcopy
import ast
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# --- Termite + constants (as in your notebook) ---
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

# ---------- helpers (exactly your notebook’s logic) ----------
def list_split_no_attrs(data, split_with=';'):
    returnable_entities = []
    if isinstance(data, str):
        for entity_value in data.split(split_with):
            entity_returnable = deepcopy(RETURN_TYPE)
            entity_returnable[ENTITY] = entity_value
            returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]

def get_cites(args):
    data_string = args['data']
    return list_split_no_attrs(data_string.citations)

def get_cited(args):
    data_string = args['data']
    return list_split_no_attrs(data_string.references)

def get_authors_ID(args):
    data_string = args['data']
    data = data_string.s2_author_ids
    if isinstance(data, str):
        split_author_ids = data.split(';')
        authors = data_string.s2_authors
        if isinstance(authors, str):
            authors_split_values = authors.split(';')
        else:
            authors_split_values = []
        returnable_entities = []
        for entity_value, attribute in zip(split_author_ids, authors_split_values):
            entity_returnable = deepcopy(RETURN_TYPE)
            entity_returnable[ENTITY] = entity_value
            entity_returnable[ATTRIBUTES] = [('name', attribute)]
            returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]
    

def get_parent_topic(args):
    data_string = args['data']
    returnable_entities = []
    parent_name = data_string.parent_name
    if type(parent_name) == str:
        entity_returnable = deepcopy(RETURN_TYPE)
        entity_returnable[ENTITY] = parent_name
        entity_returnable['attributes'] = [("Graph_Name",parent_name)]
        returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]
    
def get_topic_keywords(args):
    data_string = args['data']
    returnable_entities = []
    keyword_list = data_string.words
    if isinstance(keyword_list, str):
        keyword_list = ast.literal_eval(keyword_list)
    for keyword in keyword_list:
        entity_returnable = deepcopy(RETURN_TYPE)
        entity_returnable[ENTITY] = keyword
        returnable_entities.append(entity_returnable)
    if returnable_entities:
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]

def get_affiliations(args):
    returnable_entities = []
    affil_string =  args['data'].affiliations
    if type(affil_string) == str and affil_string != 'nan':
        for k, v in ast.literal_eval(affil_string).items():
            if isinstance(v, dict):
                entity_returnable = deepcopy(RETURN_TYPE)
                entity_returnable[ENTITY] = k
                name = v['name']
                entity_returnable[ATTRIBUTES] = [('name', name)]
                returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]
    
def get_countries(args):
    data_string = args['data']
    returnable_entities = []
    affil_string = data_string.affiliations
    if type(affil_string) == str and affil_string != 'nan':
        for k, v in ast.literal_eval(data_string.affiliations).items():
            if isinstance(v, dict):
                entity_returnable = deepcopy(RETURN_TYPE)
                entity_returnable[ENTITY] = v['country']
                returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]

def get_categories(args):
    data_string = args['data']
    returnable_entities = []
    subject_areas = data_string.subject_areas
    if type(subject_areas) == str:
        split_subjects= subject_areas.split(';')
        for subject  in split_subjects:
            entity_returnable = deepcopy(RETURN_TYPE)
            entity_returnable[ENTITY] = subject
            returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]

def split_string(args, split_with= ';'):
    data_string = args['data']
    return data_string.split(split_with)

def list_split_no_attrs(data, split_with=';'):
    returnable_entities = []
    if type(data) == str:
        split_values = data.split(split_with)
        for entity_value in split_values:
            entity_returnable = deepcopy(RETURN_TYPE)
            entity_returnable[ENTITY] = entity_value.strip()
            returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]

def get_authors_ID(args):
    data_string = args['data']
    data = data_string.author_ids
    if type(data) == str:
        split_author_ids= data.split(';')
        authors = data_string.authors
        if type(authors) == str:
            authors_split_values = authors.split(';')
        returnable_entities = []
        for entity_value, attribute in zip(split_author_ids, authors_split_values ):
            entity_returnable = deepcopy(RETURN_TYPE)
            entity_returnable[ENTITY] = entity_value
            entity_returnable[ATTRIBUTES] = [('name', attribute)]
            returnable_entities.append(entity_returnable)
        return returnable_entities
    else:
        return [deepcopy(RETURN_TYPE)]

# def get_acronyms(args):
#     data_string = args['data']
#     return list_split_no_attrs(data_string.acronym_attribution, split_with=', ')
def get_acronyms(args):
    """
    Extract acronym strings from a row, tolerating missing columns.
    Tries columns in order: 'acronym_attribution', 'acronyms', 'acronym'.
    Returns [] when nothing is present so no triples are created.
    """
    row = args.get('data', None)
    if row is None:
        return []

    candidates = ('acronym_attribution', 'acronyms', 'acronym')

    def _get_from_series(r, key):
        try:
            # pandas Series: prefer dict-style to avoid AttributeError when missing
            if hasattr(r, 'get'):
                return r.get(key, None)
            # fallback for objects with attributes
            return getattr(r, key, None)
        except Exception:
            return None

    value = None
    for col in candidates:
        v = _get_from_series(row, col)
        if v is not None and str(v).strip() and str(v).lower() != 'nan':
            value = v
            break

    if not value:
        return []  # no acronyms -> no triples

    # Accept either comma- or semicolon-separated values; normalize to commas first
    text = str(value).replace(';', ',')
    return list_split_no_attrs(text, split_with=',')



def default_topic_triplet_map():
    topics_triplet_map_keywords =  {
        'ENTITIES':[
            {ET:TOPIC_TYPE, MAKE_ID_UNIQUE:True, FROM_COL: 'Graph_Name', 
                ATTR_COL:[
                    {FROM_COL: 'label', ATTR_NAME:'label', },
                    {FROM_COL: 'Graph_Name', ATTR_NAME:'Graph_Name', }
                ]
            },
            {ET:KEYWORD_TYPE, MAKE_ID_UNIQUE:True},
            ],  
        'RELATIONS':[
            {HT:TOPIC_TYPE, R:'child_of', TT:TOPIC_TYPE, EXTRACT_T: get_parent_topic, },
            {HT:TOPIC_TYPE, R:'mentions', TT:KEYWORD_TYPE, EXTRACT_T: get_topic_keywords, },
        ]
    }
    return topics_triplet_map_keywords

def default_data_triplet_map():
    data_triplet_map_keywords =  {
        'ENTITIES':[
            {ET:TOPIC_TYPE, MAKE_ID_UNIQUE:True, FROM_COL: 'Graph_Name'},
            {ET:DOCUMENT_TYPE, FROM_COL: "doi",
                ATTR_COL:[
                    {FROM_COL: 'title', ATTR_NAME:'Title', },
                    {FROM_COL: 'eid',   ATTR_NAME:'EID', },
                    {FROM_COL: 's2id',  ATTR_NAME:'S2ID', },
                    {FROM_COL: 'doi',   ATTR_NAME:'DOI', },
                ],
                MAKE_ID_UNIQUE:True
            }, 
            {ET:AFFILIATION_IDENTIFIER_TYPE, MAKE_ID_UNIQUE:True},
            {ET:COUNTRY_TYPE,                MAKE_ID_UNIQUE:True},
            # {ET:DOCUMENT_TYPE_SCOPUS,      MAKE_ID_UNIQUE:True},
            {ET:CATEGORY, MAKE_ID_UNIQUE:True},
            {ET:ACRONYM,  MAKE_ID_UNIQUE:True},
            {ET:YEAR_TYPE, FROM_COL: 'year', ATTR_COL:None, ATTR_FUNC:None, MAKE_ID_UNIQUE:True},
            {ET:AUTHOR_ID_TYPE, FROM_COL: 'author_ids',
                ATTR_COL:[{FROM_COL: 'authors', ATTR_NAME:'Author_Name', RETREIVAL: split_string, ARGS: None}],
                ATTR_FUNC:split_string, ARGS: None, MAKE_ID_UNIQUE:True
            },
            {ET:PUBLISHER, FROM_COL: 'publication_name', MAKE_ID_UNIQUE:True},
        ],
        'RELATIONS':[
            {HT:DOCUMENT_TYPE, R:'part_of_topic', TT:TOPIC_TYPE},
            {HT:DOCUMENT_TYPE, R:DOCUMENT_YEAR_RELATION, TT:YEAR_TYPE},
            # {HT:DOCUMENT_TYPE, R:DOCUMENT_TOPIC_RELATION, TT:TOPIC_TYPE, EXTRACT_T: get_absolute_cluster,},
            {HT:AUTHOR_ID_TYPE, R:AUTHOR_DOCUMENT_RELATION, TT:DOCUMENT_TYPE, EXTRACT_H: get_authors_ID},
            {HT:DOCUMENT_TYPE,  R:DOCUMENT_AFFILITATION_RELATION, TT:AFFILIATION_IDENTIFIER_TYPE, EXTRACT_T: get_affiliations},
            {HT:AFFILIATION_IDENTIFIER_TYPE, R:AFFILIATION_COUNTRY_RELATION, TT:COUNTRY_TYPE, EXTRACT_H: get_affiliations, EXTRACT_T: get_countries, PAIRING: INDEX_PAIRING},
            # {HT:DOCUMENT_TYPE, R:DOCUMENT_CITES_RELATION, TT:DOCUMENT_TYPE_SCOPUS, EXTRACT_T: get_cites, PAIRING: HEAD_TO_MANY},
            # {HT:DOCUMENT_TYPE, R:DOCUMENT_CITED_RELATION, TT:DOCUMENT_TYPE_SCOPUS, EXTRACT_T: get_cited, PAIRING: HEAD_TO_MANY},
            {HT:DOCUMENT_TYPE, R:DOCUMENT_PUBLISHER_RELATION, TT:PUBLISHER},
            {HT:DOCUMENT_TYPE, R:DOCUMENT_CATEGORY_RELATION,  TT:CATEGORY, EXTRACT_T: get_categories, PAIRING: HEAD_TO_MANY},
            {HT:DOCUMENT_TYPE, R:DOCUMENT_ACRONYM_RELATION,   TT:ACRONYM,  EXTRACT_T: get_acronyms},
        ]
    }
    return data_triplet_map_keywords

# ---------- block with defaults ----------
DEFAULT_CALL_SETTINGS: Dict[str, Any] = {
    # Source CSVs (None → try bundle keys)
    "raw_csv_path": None,          # data/docs
    "topic_csv_path": None,        # topics/labels (NEW)

    # Output files
    "triplets_filename": "triplets.csv",      # legacy: data triplets filename
    "data_triplets_filename": None,           # NEW (falls back to triplets_filename)
    "topic_triplets_filename": "topic_triplets.csv",  # NEW

    # Column → triplet mappings
    "column_triplet_map": None,            # legacy: data map
    "column_triplet_map_data": None,       # NEW (falls back to column_triplet_map or default_data_triplet_map())
    "column_triplet_map_topics": None,     # NEW (falls back to default_topic_triplet_map())

    # Neo4j creds: env → fallback to local dev defaults
    "neo4j_uri": os.getenv("NEO4J_URI", "neo4j://localhost:7666"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_pass": os.getenv("NEO4J_PASS", "local_password"),
    # Optional auth/token passthrough for Termite
    "token": None,
}

class TermiteNeo4jBlock(AnimalBlock):
    """
    Wrapper that:
      1) builds *data* triplets from a CSV and pushes to Neo4j
      2) builds *topic* triplets from a CSV and pushes to Neo4j
    """

    CANONICAL_NEEDS: Tuple[str, ...] = ("leaf_data_csv", "leaf_labels_csv")

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("data_triplets_csv", "topic_triplets_csv", ),
        tag: str = "TermiteNeo4j",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        # Merge provided call_settings over defaults
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
        """Try bundle.get across multiple keys; return first truthy value or None."""
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
        data_csv_path = (
            self.call_settings.get("raw_csv_path")
            or self._prefer_bundle(bundle, "LeafDataLabels.leaf_data_csv", "leaf_data_csv")
        )
        if not data_csv_path:
            raise RuntimeError("[TermiteNeo4j] 'raw_csv_path' not provided and no leaf_data_csv found in bundle.")
        data_csv_path = Path(str(data_csv_path)).expanduser().resolve()

        topic_csv_path = (
            self.call_settings.get("topic_csv_path")
            or self._prefer_bundle(bundle, "LeafDataLabels.leaf_labels_csv", "leaf_labels_csv")
            or data_csv_path  # final fallback if topics are embedded in same CSV
        )
        topic_csv_path = Path(str(topic_csv_path)).expanduser().resolve()

        # ---------- Resolve outputs ----------
        data_triplets_filename = (
            self.call_settings.get("data_triplets_filename")
            or self.call_settings.get("triplets_filename", "triplets.csv")
        )
        topic_triplets_filename = self.call_settings.get("topic_triplets_filename", "topic_triplets.csv")

        data_triplets_path = out_dir / data_triplets_filename
        topic_triplets_path = out_dir / topic_triplets_filename

        # ---------- Resolve triplet maps ----------
        data_triplet_map = (
            self.call_settings.get("column_triplet_map_data")
            or self.call_settings.get("column_triplet_map")
            or default_data_triplet_map()
        )
        topic_triplet_map = self.call_settings.get("column_triplet_map_topics") or default_topic_triplet_map()

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

        # Create uniqueness constraints for both schemas up front
        termite.make_unique_constrains(data_triplet_map)
        termite.make_unique_constrains(topic_triplet_map)

        # ---------- PASS 1: DATA triplets ----------
        termite.from_csv_to_triplets(str(data_csv_path), str(data_triplets_path), data_triplet_map)
        termite.update_database_multithreaded(str(data_triplets_path))

        # ---------- PASS 2: TOPIC triplets ----------
        termite.from_csv_to_triplets(str(topic_csv_path), str(topic_triplets_path), topic_triplet_map)
        termite.update_database_multithreaded(str(topic_triplets_path))

        # ---------- Register outputs ----------
        # Back-compat: keep ".triplets_csv" pointing to the DATA triplets
        self.register_checkpoint("data_triplets_csv", data_triplets_path)
        self.register_checkpoint("topic_triplets_csv", topic_triplets_path)
        # self.register_checkpoint("triplets_csv", data_triplets_path)

        bundle[f"{self.tag}.data_triplets_csv"] = data_triplets_path
        bundle[f"{self.tag}.topic_triplets_csv"] = topic_triplets_path
        # bundle[f"{self.tag}.triplets_csv"] = data_triplets_path  # alias

        if self.verbose:
            print(f"[{self.tag}] Data triplets @ {data_triplets_path}")
            print(f"[{self.tag}] Topic triplets @ {topic_triplets_path}")
