# wolf_block.py

from __future__ import annotations
from typing import Dict, Sequence, Any, Tuple
import os
from pathlib import Path

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
    provides: ['graph']
    Automatically checkpoints 'graph' to disk as 'graph.gpickle'.
    """

    CANONICAL_NEEDS = ('df', 'map')
    WOLF_STATS = ['page_rank', 'hubs_authorities', 'betweenness_centrality']

    category_map = {
        "co-author": {
            "col":      "slic_author_ids",
            "name_col": "slic_authors",
            "png":      "all_co-authors.png",
            "ranks":    "co-author_rankings.csv",
            "html":     "co-authors.html",
        },
        "co-affiliation": {
            "col":      "affiliation_ids",
            "name_col": "affiliation_names",
            "png":      "all_co-affiliations.png",
            "ranks":    "co-affiliation_rankings.csv",
            "html":     "co-affiliations.html",
        },
        "co-country": {
            "col":      "countries",
            "name_col": "countries",
            "png":      "all_co-countries.png",
            "ranks":    "co-country_rankings.csv",
            "html":     "co-countries.html",
        },
    }

    def __init__(
        self,
        *,
        category: str = 'co-author',
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ('graph',),
        tag: str = "Wolf",
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        verbose: bool = True,
    ) -> None:
        if category not in self.category_map:
            raise ValueError(f"Unknown category {category!r}")

        self.category = category

        # allow multiple WolfBlock instances without key collision
        if provides == ('graph',):
            provides = (f"graph_{self.category}",)

        default_init = {"verbose": True}
        default_call = {}

        # By default, checkpoint_keys is None → AnimalBlock will use self.provides
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
        # ─── 1) Load the DataFrame & map ──────────────────────────────────────
        df  = self.load_path(bundle[self.needs[0]])  
        print("Number of rows in df:", len(df))
        print("Unique IDs:", df[self.category_map[self.category]['col']].nunique())

        orca_map = bundle[self.needs[1]]

        OUTPUT_DIR = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        output_dir = Path(check_path(os.path.join(OUTPUT_DIR, self.category)))

        # ─── 2) Build the codependency matrix ────────────────────────────────
        codep = CodependencyMatrixBlock(
            col=self.category_map[self.category]['col']
        )
        sub_bundle = DataBundle({
            'df': df,
            SAVE_DIR_BUNDLE_KEY: OUTPUT_DIR
        })
        codep(sub_bundle)
        X, node_ids = (
            sub_bundle[codep.provides[0]],
            sub_bundle[codep.provides[1]]
        )

        # ─── 3) Prepare node attributes ──────────────────────────────────────
        wolf = Wolf(**self.init_settings)
        wolf.node_ids  = node_ids

        if self.category == "co-author":
            wolf.attributes = create_attributes(orca_map, attribute_names=[])
 

        elif self.category == "co-affiliation":
            name_col = self.category_map[self.category]['name_col']
            id_col   = self.category_map[self.category]['col']
            mapping  = get_id_to_name(df, name_col, id_col)
            wolf.attributes = {k: {'name': v} for k, v in mapping.items()}
 
        # ─── 4) Create & annotate the graph ───────────────────────────────────
        graph = wolf.create_graph(X, use_weighted_value=True)
        for stat in tqdm(self.WOLF_STATS):
            graph.get_stat(stat)

        # ─── 5) Output rankings CSV ──────────────────────────────────────────
        stats_df = graph.output_stats()
        numeric = stats_df.select_dtypes(include=[np.number]).columns
        stats_df[numeric] = stats_df[numeric].map(apply_alpha)
        stats_df = stats_df \
            .sort_values(by=next(iter(self.WOLF_STATS)), ascending=False) \
            .reset_index(drop=True)

        stats_df.to_csv(
            output_dir / self.category_map[self.category]['ranks'],
            index=False,
            encoding="utf-8-sig"
        )

        # ─── 6) Save full-network plot & HTML ────────────────────────────────
        graph.visualize(
            font_color      = 'black',
            node_color      = '#edede9',
            node_size       = 100,
            highlight_nodes = [],
            font_size       = 4,
            edge_width      = 0.08,
            figsize         = (8, 8),
            save_path       = str(output_dir / self.category_map[self.category]['png'])
        )

        fig = plot_authors_graph(
            df     = df,
            id_col = self.category_map[self.category]['col'],
            name_col = self.category_map[self.category]['name_col'],
        )
        fig.write_html(str(output_dir / self.category_map[self.category]['html']))

        # ─── 7) Component subplots & word-clouds ─────────────────────────────
        save_components(
            df          = df,
            ranking_df  = stats_df,
            g           = graph,
            col         = self.category_map[self.category]['col'],
            results_dir = str(output_dir),
        )
        component_wordclouds(
            df          = df,
            g           = graph,
            col         = self.category_map[self.category]['col'],
            results_dir = str(output_dir),
        )

        # ─── 8) Checkpoint the graph ─────────────────────────────────────────
        graph_path = output_dir / "graph.gpickle"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path(graph, graph_path)

        # Tell AnimalBlock to record this file under the key "graph"
        self.register_checkpoint(self.provides[0], graph_path)
        # Finally, put the graph into the bundle under your namespaced key
        bundle[f"{self.tag}.{self.provides[0]}"] = graph