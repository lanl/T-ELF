# pipeline/blocks/peacock_stats_block.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence, Optional, Tuple

import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# Peacock aggregation function
from ...post_processing.Peacock.Utility import aggregate_ostats

# Peacock plotting functions (the ones you pasted from Plot/plot.py)
from ...post_processing.Peacock.Plot import (
    plot_heatmap,
    plot_bar,
    plot_scatter,
)

# Convenience aliases
plot_hist      = plot_bar
plot_scatter3D = plot_scatter


class PeacockStatsBlock(AnimalBlock):
    CANONICAL_NEEDS: Tuple[str, ...] = ("df", SAVE_DIR_BUNDLE_KEY)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("outpath",),
        hist_stats: Sequence[str] = ("paper_count", "num_citations"),
        hist_ylabels: Optional[Dict[str, str]] = None,
        col_names: Optional[Dict[str, str]] = None,
        affiliation_palette: Optional[Dict[str, str]] = None,
        country: Optional[str] = None,
        **kw: Any,
    ) -> None:
        super().__init__(
            needs=needs,
            provides=provides,
            tag="PeacockStats",
            init_settings={},
            call_settings={},
            **kw,
        )
        self.hist_stats = tuple(hist_stats)
        self.hist_ylabels = hist_ylabels or {
            "paper_count": "Number of Papers",
            "num_citations": "Number of Citations",
            "attribution_percentage": "Attribution Percentage",
        }
        self.col_names = col_names or {
            "id":           "eid",
            "authors":      "slic_authors",
            "author_ids":   "slic_author_ids",
            "affiliations": "slic_affiliations",
            "funding":      "funding",
            "citations":    "num_citations",
            "references":   "references",
        }
        self.affiliation_palette = affiliation_palette or {}
        self.country = country

    def run(self, bundle: DataBundle) -> None:
        # 1) Inputs & cleanup
        df: pd.DataFrame = bundle["df"]
        # Save everything under the block's tag directory (like other blocks)
        root_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY])
        out_dir = root_dir / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # ensure affiliation column is string
        aff_col = self.col_names["affiliations"]
        df[aff_col] = df[aff_col].apply(lambda x: x if isinstance(x, str) else str(x))

        df = (
            df
            .dropna(subset=[
                self.col_names["id"],
                self.col_names["authors"],
                self.col_names["author_ids"],
                aff_col,
            ])
            .assign(year=lambda d: d.year.astype(int))
        )

        filters = {"country": self.country} if self.country else None

        # 2) Write top‐100 CSVs
        author_stats      = aggregate_ostats(df, key="author_id",       top_n=100, col_names=self.col_names, filters=filters, by_year=False)
        affiliation_stats = aggregate_ostats(df, key="affiliation_id", top_n=100, col_names=self.col_names, filters=filters, by_year=False)
        author_stats.to_csv(out_dir / "top_authors.csv",      index=False)
        affiliation_stats.to_csv(out_dir / "top_affiliations.csv", index=False)

        # 3) Shared “top‐10 by citations” args
        auth_args = dict(
            key="author_id",
            top_n=10,
            sort_by="num_citations",
            col_names=self.col_names,
            by_year=True,
            filters=filters,
        )
        aff_args = dict(
            key="affiliation_id",
            top_n=10,
            sort_by="num_citations",
            col_names=self.col_names,
            by_year=True,
            filters=filters,
        )

        # 4) Heatmaps — pivot then call plot_heatmap
        # ─ Authors, citations
        auth_heat = aggregate_ostats(df, **auth_args)
        pivot_c   = auth_heat.pivot(index="year", columns="author", values="num_citations").fillna(0)
        plot_heatmap(
            pivot_c,
            cmap="jet",
            interpolation="gaussian",
            fname=str(out_dir / "author_heatmap_citations.png"),
            interactive=False,
            title="Author Citations by Year",
            xlabel="Author",
            ylabel="Year",
        )
        # ─ Authors, papers
        pivot_p = auth_heat.pivot(index="year", columns="author", values="paper_count").fillna(0)
        plot_heatmap(
            pivot_p,
            cmap="jet",
            interpolation="gaussian",
            fname=str(out_dir / "author_heatmap_papers.png"),
            interactive=False,
            title="Author Papers by Year",
            xlabel="Author",
            ylabel="Year",
        )

        # ─ Affiliations, citations
        aff_heat = aggregate_ostats(df, **aff_args)
        pivot_c2 = aff_heat.pivot(index="year", columns="affiliation", values="num_citations").fillna(0)
        plot_heatmap(
            pivot_c2,
            cmap="jet",
            interpolation="gaussian",
            fname=str(out_dir / "affiliation_heatmap_citations.png"),
            interactive=False,
            title="Affiliation Citations by Year",
            xlabel="Affiliation",
            ylabel="Year",
        )
        # ─ Affiliations, papers
        pivot_p2 = aff_heat.pivot(index="year", columns="affiliation", values="paper_count").fillna(0)
        plot_heatmap(
            pivot_p2,
            cmap="jet",
            interpolation="gaussian",
            fname=str(out_dir / "affiliation_heatmap_papers.png"),
            interactive=False,
            title="Affiliation Papers by Year",
            xlabel="Affiliation",
            ylabel="Year",
        )

        # 5) Histograms (bar‐plots)
        auth_hist = aggregate_ostats(df, **{**auth_args, "by_year": False})
        plot_hist(
            auth_hist,
            x="author",
            ys=list(self.hist_stats),
            fname=str(out_dir / "author_hist.png"),
            interactive=False,
            # cmap="husl",                     # ← remove this line...
            title="Author Statistics Histogram",
            xlabel="Author",
            ylabel=self.hist_ylabels[self.hist_stats[0]],
        )
        aff_hist = aggregate_ostats(df, **{**aff_args, "by_year": False})
        plot_hist(
            aff_hist,
            x="affiliation",
            ys=list(self.hist_stats),
            fname=str(out_dir / "affiliation_hist.png"),
            interactive=False,
            # cmap="husl",
            title="Affiliation Statistics Histogram",
            xlabel="Affiliation",
            ylabel=self.hist_ylabels[self.hist_stats[0]],
        )

        # 6) 3-D scatter
        plot_scatter3D(
            df,
            x="paper_count",
            y="attribution_percentage",
            z="num_citations",
            agg_func=aggregate_ostats,
            agg_kwargs=auth_args,
            fname=str(out_dir / "author_scatter.png"),
            interactive=False,
            log_z=True,
            hue="affiliation",
            labels="author",
            title="Author Stats Scatter3D",
            xlabel="Paper Count",
            ylabel="Attribution Percentage",
            zlabel="Num. Citations",
            base_palette=self.affiliation_palette,
        )
        plot_scatter3D(
            df,
            x="paper_count",
            y="attribution_percentage",
            z="num_citations",
            agg_func=aggregate_ostats,
            agg_kwargs=aff_args,
            fname=str(out_dir / "affiliation_scatter.png"),
            interactive=False,
            log_z=True,
            hue="country",
            labels="affiliation",
            title="Affiliation Stats Scatter3D",
            xlabel="Paper Count",
            ylabel="Attribution Percentage",
            zlabel="Num. Citations",
            base_palette=self.affiliation_palette,
        )

        # 7) Dummy checkpoint under your single `provides` key
        if SAVE_DIR_BUNDLE_KEY in bundle:
            final_csv = out_dir / "none.csv"
            final_csv.write_text("")  # empty placeholder
            self.register_checkpoint(self.provides[0], final_csv)
            bundle[f"{self.tag}.{self.provides[0]}"] = final_csv
        # (no return: AnimalBlock.__call__ returns the bundle)
