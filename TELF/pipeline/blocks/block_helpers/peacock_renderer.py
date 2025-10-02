# post_processing/Peacock/peacock_renderer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence
import ast
import json
import numpy as np
import pandas as pd

from ....post_processing.Peacock.Utility import aggregate_ostats
from ....post_processing.Peacock.Plot import plot_heatmap, plot_bar, plot_scatter

plot_hist      = plot_bar
plot_scatter3D = plot_scatter


def _normalize_aff_to_py_literal(v):
    """
    Accept dict/list/JSON/string; return a *Python-literal* string (repr),
    guaranteeing per-affiliation 'authors' (list) & 'country' (str).
    """
    if isinstance(v, (dict, list)):
        obj = v
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            obj = []
        else:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                try:
                    obj = json.loads(s)
                except Exception:
                    obj = []
    else:
        obj = []

    if isinstance(obj, list):
        out = {}
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                continue
            key = str(item.get("id", item.get("affiliation_id", i)))
            out[key] = {
                **item,
                "authors": item.get("authors", item.get("author_ids", [])) or [],
                "country": item.get("country", "Unknown"),
            }
        obj = out
    elif isinstance(obj, dict):
        out = {}
        for k, val in obj.items():
            if not isinstance(val, dict):
                continue
            val.setdefault("authors", val.get("author_ids", []))
            val.setdefault("country", "Unknown")
            out[str(k)] = val
        obj = out
    else:
        obj = {}

    return repr(obj)


class PeacockRenderer:
    def __init__(
        self,
        *,
        hist_stats: Sequence[str] = ("paper_count", "num_citations"),
        hist_ylabels: Optional[Dict[str, str]] = None,
        col_names: Optional[Dict[str, str]] = None,
        affiliation_palette: Optional[Dict[str, str]] = None,
        country: Optional[str] = None,
        cluster_col: Optional[str] = "cluster",   # NEW
    ) -> None:
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
        self.cluster_col = cluster_col

    def _png_or_html(self, make_plot_func, stem: Path, *args, **kwargs):
        png_path  = stem.with_suffix(".png")
        html_path = stem.with_suffix(".html")
        try:
            fig = make_plot_func(*args, interactive=True, fname=None, **kwargs)
            fig.write_html(str(html_path), include_plotlyjs="cdn")
        except Exception:
            print("Exception making interactive plots in peacock")
            make_plot_func(*args, interactive=False, fname=str(png_path), **kwargs)

    def render(self, df: pd.DataFrame, out_dir: Path) -> None:
        # run overall
        self._render_core(df, out_dir)

        # run per-cluster
        if self.cluster_col and self.cluster_col in df.columns:
            cluster_root = out_dir / "clusters"
            for cid, df_c in df.groupby(self.cluster_col, dropna=False):
                safe = "nan" if pd.isna(cid) else str(cid).replace("/", "_")
                self._render_core(df_c, cluster_root / safe)

    def _render_core(self, df: pd.DataFrame, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()

        aff_col = self.col_names["affiliations"]
        aut_col = self.col_names["authors"]
        aid_col = self.col_names["author_ids"]

        def _to_list_any(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, str):
                s = x.strip()
                if not s:
                    return []
                for parser in (ast.literal_eval, json.loads):
                    try:
                        v = parser(s)
                        if isinstance(v, list):
                            return v
                    except Exception:
                        pass
                sep = ";" if ";" in s else ","
                return [t.strip() for t in s.split(sep) if t.strip()]
            return [str(x)]

        def _to_sc_str_preserve_nan(x):
            lst = _to_list_any(x)
            vals = [str(v).strip() for v in lst if str(v).strip() != ""]
            return np.nan if not vals else ";".join(vals)

        df[aut_col] = df[aut_col].apply(_to_sc_str_preserve_nan)
        df[aid_col] = df[aid_col].apply(_to_sc_str_preserve_nan)

        subset = [self.col_names["id"], aut_col, aid_col, aff_col]
        df = df.dropna(subset=subset)

        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)
        else:
            df["year"] = 0

        if df.empty:
            (out_dir / "top_authors.csv").write_text("")
            (out_dir / "top_affiliations.csv").write_text("")
            return

        df[aff_col] = df[aff_col].apply(_normalize_aff_to_py_literal)

        filters = {"country": self.country} if self.country else None

        def _safe_pivot_table(data, index, columns, values):
            if data is None or len(data) == 0:
                return pd.DataFrame()
            return data.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)

        author_stats      = aggregate_ostats(df, key="author_id",       top_n=100, col_names=self.col_names, filters=filters, by_year=False)
        affiliation_stats = aggregate_ostats(df, key="affiliation_id", top_n=100, col_names=self.col_names, filters=filters, by_year=False)
        author_stats.to_csv(out_dir / "top_authors.csv", index=False)
        affiliation_stats.to_csv(out_dir / "top_affiliations.csv", index=False)

        auth_args = dict(key="author_id",       top_n=10, sort_by="num_citations", col_names=self.col_names, by_year=True,  filters=filters)
        aff_args  = dict(key="affiliation_id",  top_n=10, sort_by="num_citations", col_names=self.col_names, by_year=True,  filters=filters)

        auth_heat = aggregate_ostats(df, **auth_args)
        aff_heat  = aggregate_ostats(df, **aff_args)

        pivot_c  = _safe_pivot_table(auth_heat, index="year", columns="author",      values="num_citations")
        pivot_p  = _safe_pivot_table(auth_heat, index="year", columns="author",      values="paper_count")
        pivot_c2 = _safe_pivot_table(aff_heat,  index="year", columns="affiliation", values="num_citations")
        pivot_p2 = _safe_pivot_table(aff_heat,  index="year", columns="affiliation", values="paper_count")

        if not pivot_c.empty:
            self._png_or_html(plot_heatmap, out_dir / "author_heatmap_citations",
                pivot_c, cmap="jet", interpolation="gaussian",
                title="Author Citations by Year", xlabel="Author", ylabel="Year")
        if not pivot_p.empty:
            self._png_or_html(plot_heatmap, out_dir / "author_heatmap_papers",
                pivot_p, cmap="jet", interpolation="gaussian",
                title="Author Papers by Year", xlabel="Author", ylabel="Year")
        if not pivot_c2.empty:
            self._png_or_html(plot_heatmap, out_dir / "affiliation_heatmap_citations",
                pivot_c2, cmap="jet", interpolation="gaussian",
                title="Affiliation Citations by Year", xlabel="Affiliation", ylabel="Year")
        if not pivot_p2.empty:
            self._png_or_html(plot_heatmap, out_dir / "affiliation_heatmap_papers",
                pivot_p2, cmap="jet", interpolation="gaussian",
                title="Affiliation Papers by Year", xlabel="Affiliation", ylabel="Year")

        auth_hist = aggregate_ostats(df, **{**auth_args, "by_year": False})
        if not auth_hist.empty:
            self._png_or_html(plot_hist, out_dir / "author_hist",
                auth_hist, x="author", ys=list(self.hist_stats),
                title="Author Statistics Histogram", xlabel="Author",
                ylabel=self.hist_ylabels[self.hist_stats[0]])

        aff_hist = aggregate_ostats(df, **{**aff_args, "by_year": False})
        if not aff_hist.empty:
            self._png_or_html(plot_hist, out_dir / "affiliation_hist",
                aff_hist, x="affiliation", ys=list(self.hist_stats),
                title="Affiliation Statistics Histogram", xlabel="Affiliation",
                ylabel=self.hist_ylabels[self.hist_stats[0]])

        self._png_or_html(plot_scatter3D, out_dir / "author_scatter",
            df, x="paper_count", y="attribution_percentage", z="num_citations",
            agg_func=aggregate_ostats, agg_kwargs=auth_args,
            log_z=True, hue="affiliation", labels="author",
            title="Author Stats Scatter3D", xlabel="Paper Count",
            ylabel="Attribution Percentage", zlabel="Num. Citations",
            base_palette=self.affiliation_palette)

        self._png_or_html(plot_scatter3D, out_dir / "affiliation_scatter",
            df, x="paper_count", y="attribution_percentage", z="num_citations",
            agg_func=aggregate_ostats, agg_kwargs=aff_args,
            log_z=True, hue="country", labels="affiliation",
            title="Affiliation Stats Scatter3D", xlabel="Paper Count",
            ylabel="Attribution Percentage", zlabel="Num. Citations",
            base_palette=self.affiliation_palette)
