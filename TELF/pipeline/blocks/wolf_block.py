# TELF/pipeline/blocks/wolf_block.py
from __future__ import annotations
from typing import Dict, Sequence, Any, Tuple
import os
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...post_processing import Wolf
from ...post_processing.Wolf.utils import create_attributes
from ...post_processing.Wolf.plots import component_wordclouds, save_components

from ...helpers.file_system import check_path
from ...helpers.figures import plot_authors_graph
from ...helpers.frames import apply_alpha
from ...helpers.maps import get_id_to_name

from .beaver_codependency_matrix_block import CodependencyMatrixBlock


class WolfBlock(AnimalBlock):
    """
    needs:    ['df', 'map']
    provides: ['graph_<category>']
    Automatically checkpoints 'graph' to disk as 'graph.gpickle'.
    """

    CANONICAL_NEEDS = ("df", "map")
    WOLF_STATS = ["page_rank", "hubs_authorities", "betweenness_centrality"]

    category_map = {
        "co-author": {
            "col": "slic_author_ids",
            "name_col": "slic_authors",
            "png": "all_co-authors.png",
            "ranks": "co-author_rankings.csv",
            "html": "co-authors.html",
        },
        "co-affiliation": {
            "col": "affiliation_ids",
            "name_col": "affiliation_names",
            "png": "all_co-affiliations.png",
            "ranks": "co-affiliation_rankings.csv",
            "html": "co-affiliations.html",
        },
        "co-country": {
            "col": "countries",
            "name_col": "countries",
            "png": "all_co-countries.png",
            "ranks": "co-country_rankings.csv",
            "html": "co-countries.html",
        },
    }

    def __init__(
        self,
        *,
        category: str = "co-author",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("graph",),
        tag: str = "Wolf",
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        if category not in self.category_map:
            raise ValueError(f"Unknown category {category!r}")

        self.category = category

        if provides == ("graph",):
            provides = (f"graph_{self.category}",)

        default_init = {"verbose": True}
        default_call: Dict[str, Any] = {}

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings={**default_init, **(init_settings or {})},
            call_settings={**default_call, **(call_settings or {})},
            verbose=verbose,
        )

    # -----------------------
    # Helpers
    # -----------------------
    def _normalize_ids_series(self, s: pd.Series) -> pd.Series:
        """
        Normalize an ID column to semicolon-delimited strings.

        - Accept lists/tuples/sets and join with ';'
        - Convert common delimiters (',', '|', tab) to ';'
        - Trim spaces, drop empties
        """
        def _norm(v):
            if pd.isna(v):
                return ""
            if isinstance(v, (list, tuple, set)):
                v = ";".join(map(str, v))
            else:
                v = str(v)
            for sep in [",", "|", "\t"]:
                v = v.replace(sep, ";")
            parts = [p.strip() for p in v.split(";") if p and p.strip()]
            return ";".join(parts)

        return s.map(_norm)

    def _write_empty_artifacts(self, output_dir: Path) -> nx.Graph:
        """Create empty outputs (CSV + graph) and return empty graph."""
        stats_path = output_dir / self.category_map[self.category]["ranks"]
        pd.DataFrame(columns=["node", *self.WOLF_STATS]).to_csv(
            stats_path, index=False, encoding="utf-8-sig"
        )
        g = nx.Graph()
        graph_path = output_dir / "graph.gpickle"
        self.save_path(g, graph_path)
        self.register_checkpoint(self.provides[0], graph_path)
        return g

    def _build_codep_matrix_fallback(self, s: pd.Series) -> tuple[np.ndarray, list[str]]:
        """
        Build a simple symmetric co-occurrence matrix from a normalized
        semicolon-delimited ID series.
        """
        ids_per_row = [
            [tok for tok in str(v).split(";") if tok]
            for v in s.fillna("")
        ]
        # collect nodes
        nodes = []
        seen = set()
        for row in ids_per_row:
            for tok in row:
                if tok not in seen:
                    seen.add(tok)
                    nodes.append(tok)

        n = len(nodes)
        if n < 2:
            return np.zeros((0, 0), dtype=float), []

        idx = {node: i for i, node in enumerate(nodes)}
        X = np.zeros((n, n), dtype=float)

        # count co-occurrences
        for row in ids_per_row:
            unique_row = sorted(set(row))
            for a, b in combinations(unique_row, 2):
                i, j = idx[a], idx[b]
                X[i, j] += 1.0
                X[j, i] += 1.0

        return X, nodes

    def _coerce_node_ids_for_wolf(self, node_ids_any) -> dict | list | tuple | None:
        """
        Coerce various node_ids shapes into what Wolf expects:
        - dict with sequential-int keys -> OK
        - list/tuple of two dicts       -> OK (bipartite)
        - list/array of labels          -> convert to {i: label}
        - pandas Index/Series           -> convert to {i: label}
        - None                          -> OK
        """
        if node_ids_any is None:
            return None

        # Already a dict (unipartite)
        if isinstance(node_ids_any, dict):
            return node_ids_any

        # 2-part structure from some codep implementations
        if isinstance(node_ids_any, (list, tuple)) and len(node_ids_any) == 2 \
           and all(isinstance(d, dict) for d in node_ids_any):
            return node_ids_any

        # List/array/index/series of labels -> make {i: label}
        if isinstance(node_ids_any, (list, tuple, np.ndarray, pd.Index, pd.Series)):
            labels = list(node_ids_any)
            labels = [str(x) for x in labels]
            return {i: lab for i, lab in enumerate(labels)}

        # Fallback: single value?
        try:
            return {0: str(node_ids_any)}
        except Exception:
            raise TypeError(
                "Unsupported node_ids type; expected dict, (dict, dict), or a sequence of labels."
            )

    # -----------------------
    # Main
    # -----------------------
    def run(self, bundle: DataBundle) -> None:
        # 1) Load inputs
        df = self.load_path(bundle[self.needs[0]])
        orca_map = bundle[self.needs[1]]

        print("Number of rows in df:", len(df))
        ids_col = self.category_map[self.category]["col"]
        if isinstance(df, pd.DataFrame) and ids_col in df.columns:
            try:
                print("Unique raw values in ID column:", df[ids_col].nunique())
            except Exception:
                print("Unique raw values in ID column: (undetermined)")
        else:
            print(f"Unique IDs: 0 (column '{ids_col}' missing)")

        OUTPUT_ROOT = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        output_dir = Path(check_path(os.path.join(OUTPUT_ROOT, self.category)))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guards: 'year' & minimum nodes
        if "year" not in df.columns or df["year"].isna().all():
            df = df.copy()
            df["year"] = 0

        # Normalize ID column if present
        if ids_col in df.columns:
            df = df.copy()
            df[ids_col] = self._normalize_ids_series(df[ids_col])
            series = df[ids_col]
        else:
            # Use a SAFE Series default if the column is missing
            series = pd.Series([], dtype=object)

        # Count distinct nodes after normalization
        nodes_count = (
            series.dropna()
            .astype(str)
            .str.split(";")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
        print(f"Usable unique node count after normalization: {nodes_count}")

        if nodes_count < 2:
            # produce empty artifacts & exit gracefully
            g = self._write_empty_artifacts(output_dir)
            bundle[f"{self.tag}.{self.provides[0]}"] = g
            print(f"[Wolf/{self.category}] Skipped: only {nodes_count} unique node(s) or column missing.")
            return

        # 2) Co-dependency matrix
        X, node_ids = None, None
        try:
            codep = CodependencyMatrixBlock(
                col=ids_col,
                call_settings={
                    "split_authors_with": ";",  # normalized to ';'
                    "n_jobs": 1,                # robust chunking
                },
            )
            sub_bundle = DataBundle({"df": df, SAVE_DIR_BUNDLE_KEY: OUTPUT_ROOT})
            codep(sub_bundle)

            expected = getattr(codep, "provides", ("X", "node_ids"))
            if all(k in sub_bundle for k in expected):
                X, node_ids = (sub_bundle[expected[0]], sub_bundle[expected[1]])
            else:
                print(f"[Wolf/{self.category}] BeaverCodependencyMatrix missing outputs {list(expected)}; using fallback.")
        except Exception as e:
            print(f"[Wolf/{self.category}] BeaverCodependencyMatrix failed with {type(e).__name__}: {e}")
            # fall back below

        # Fallback path if Beaver failed or didn't provide outputs
        if X is None or node_ids is None:
            X, nodes = self._build_codep_matrix_fallback(series)
            if len(nodes) < 2:
                g = self._write_empty_artifacts(output_dir)
                bundle[f"{self.tag}.{self.provides[0]}"] = g
                print(f"[Wolf/{self.category}] Skipped: fallback produced <2 nodes.")
                return
            node_ids = nodes  # list -> will be coerced below

        # ---- COERCE node_ids into the shape Wolf expects ----
        node_ids = self._coerce_node_ids_for_wolf(node_ids)

        # 3) Node attributes
        wolf = Wolf(**self.init_settings)
        wolf.node_ids = node_ids  # now valid types only

        if self.category == "co-author":
            wolf.attributes = create_attributes(orca_map, attribute_names=[])
        elif self.category == "co-affiliation":
            name_col = self.category_map[self.category]["name_col"]
            id_col = self.category_map[self.category]["col"]
            if name_col in df.columns and id_col in df.columns:
                mapping = get_id_to_name(df, name_col, id_col)
                wolf.attributes = {k: {"name": v} for k, v in mapping.items()}
            else:
                wolf.attributes = {}
        else:
            wolf.attributes = {}

        # 4) Create graph & stats
        graph = wolf.create_graph(X, use_weighted_value=True)
        for stat in tqdm(self.WOLF_STATS):
            graph.get_stat(stat)

        stats_df = graph.output_stats()
        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df[numeric_cols] = stats_df[numeric_cols].applymap(apply_alpha)
        stats_df = stats_df.sort_values(by=self.WOLF_STATS[0], ascending=False).reset_index(drop=True)

        stats_df.to_csv(
            output_dir / self.category_map[self.category]["ranks"],
            index=False,
            encoding="utf-8-sig",
        )

        # 5) Plots
        graph.visualize(
            font_color="black",
            node_color="#edede9",
            node_size=100,
            highlight_nodes=[],
            font_size=4,
            edge_width=0.08,
            figsize=(8, 8),
            save_path=str(output_dir / self.category_map[self.category]["png"]),
        )

        fig = plot_authors_graph(
            df=df,
            id_col=self.category_map[self.category]["col"],
            name_col=self.category_map[self.category]["name_col"],
        )
        fig.write_html(str(output_dir / self.category_map[self.category]["html"]))

        # 6) Components & word-clouds
        save_components(
            df=df,
            ranking_df=stats_df,
            g=graph,
            col=self.category_map[self.category]["col"],
            results_dir=str(output_dir),
        )
        component_wordclouds(
            df=df,
            g=graph,
            col=self.category_map[self.category]["col"],
            results_dir=str(output_dir),
        )

        # 7) Checkpoint
        graph_path = output_dir / "graph.gpickle"
        self.save_path(graph, graph_path)
        self.register_checkpoint(self.provides[0], graph_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = graph
