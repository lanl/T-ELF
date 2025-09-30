# TELF/pipeline/blocks/wolf_block.py
from __future__ import annotations
from typing import Dict, Sequence, Any, Tuple
import os
from pathlib import Path

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

    def run(self, bundle: DataBundle) -> None:
        # 1) Load inputs
        df = self.load_path(bundle[self.needs[0]])
        orca_map = bundle[self.needs[1]]

        print("Number of rows in df:", len(df))
        ids_col = self.category_map[self.category]["col"]
        if isinstance(df, pd.DataFrame) and ids_col in df.columns:
            try:
                print("Unique IDs:", df[ids_col].nunique())
            except Exception:
                print("Unique IDs: (undetermined)")
        else:
            print(f"Unique IDs: 0 (column '{ids_col}' missing)")

        OUTPUT_ROOT = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        output_dir = Path(check_path(os.path.join(OUTPUT_ROOT, self.category)))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guards: 'year' & minimum nodes
        if "year" not in df.columns or df["year"].isna().all():
            df = df.copy()
            df["year"] = 0

        # Use a SAFE Series default if the column is missing
        series = df[ids_col] if ids_col in df.columns else pd.Series([], dtype=object)

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

        if nodes_count < 2:
            # produce empty artifacts & exit gracefully
            stats_path = output_dir / self.category_map[self.category]["ranks"]
            pd.DataFrame(columns=["node", *self.WOLF_STATS]).to_csv(
                stats_path, index=False, encoding="utf-8-sig"
            )

            g = nx.Graph()
            graph_path = output_dir / "graph.gpickle"
            self.save_path(g, graph_path)
            self.register_checkpoint(self.provides[0], graph_path)
            bundle[f"{self.tag}.{self.provides[0]}"] = g
            print(f"[Wolf/{self.category}] Skipped: only {nodes_count} unique node(s) or column missing.")
            return

        # 2) Co-dependency matrix
        codep = CodependencyMatrixBlock(
            col=ids_col,
            call_settings={
                "split_authors_with": ";",  # SLIC-separated ids
                "n_jobs": 1,                # robust chunking
            },
        )
        sub_bundle = DataBundle({"df": df, SAVE_DIR_BUNDLE_KEY: OUTPUT_ROOT})
        codep(sub_bundle)
        X, node_ids = (sub_bundle[codep.provides[0]], sub_bundle[codep.provides[1]])

        # 3) Node attributes
        wolf = Wolf(**self.init_settings)
        wolf.node_ids = node_ids

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
        numeric = stats_df.select_dtypes(include=[np.number]).columns
        stats_df[numeric] = stats_df[numeric].map(apply_alpha)
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
