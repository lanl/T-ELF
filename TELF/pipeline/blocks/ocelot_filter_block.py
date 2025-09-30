from pathlib import Path
from typing import Dict, Any, Optional, List
import json, pprint
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...applications import Ocelot   # <-- import your Ocelot class


class OcelotFilterBlock(AnimalBlock):
    """
    Markdown-driven Ocelot filter (per-main + global constraints), hop-aware,
    with verbose logging and artifact outputs (filtered df + explain table + terms).
    """

    CANONICAL_NEEDS = ("df", "term_path")

    def __init__(
        self,
        *,
        needs=CANONICAL_NEEDS,
        provides=("df", "ocelot_table"),
        # Optional columns to construct a text field if not present
        text_columns: Optional[Dict[str, str]] = None,
        # Default id/text column names in input df
        id_field: str = "eid",
        text_field: str = "text",
        tag: str = "OcelotFilter",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        """
        text_columns:
            {
              "title":    "title",
              "abstract": "abstract",
              "fallback_text": "title_abstract"   # if present, will be used/filled
            }
        If `text_field` doesn’t exist, we’ll build it from title+abstract or fallback_text.
        """
        self.id_field = id_field
        self.text_field = text_field
        self.text_columns = text_columns or {
            "title": "title",
            "abstract": "abstract",
            "fallback_text": "title_abstract",
        }

        # Block-level defaults
        default_init = {
            "verbose": True,
            "use_hops": False,
        }
        # Ocelot “call” settings (see Ocelot.__init__/from_markdown)
        default_call = {
            "positives_mode": "any",          # per-rule positives: "any" or "all"
            "global_positives_mode": "any",   # global positives:  "any" or "all"
            "emit_nonmatches": False,         # include fails in explain table
        }

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

        # ------------------------------------------------------ helpers

    def _rebuild_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALWAYS rebuild df[self.text_field]:
        Priority: fallback_text -> (title + abstract) -> "".
        Any pre-existing text column is ignored and dropped.
        """
        tf = self.text_field
        title = self.text_columns.get("title", "title")
        abstract = self.text_columns.get("abstract", "abstract")
        fallback = self.text_columns.get("fallback_text", "title_abstract")

        # Drop any pre-existing text column
        if tf in df.columns:
            df = df.drop(columns=[tf])
            print(f"[{self.tag}] dropped existing text field '{tf}' (forced rebuild)")

        if fallback in df.columns:
            # Use fallback directly
            df[tf] = df[fallback].fillna("").astype(str)
            print(f"[{self.tag}] rebuilt '{tf}' from fallback '{fallback}'")
            return df

        # Build from title + abstract if either exists
        has_title = title in df.columns
        has_abstract = abstract in df.columns

        if has_title or has_abstract:
            t_series = df.get(title, pd.Series("", index=df.index)).fillna("").astype(str)
            a_series = df.get(abstract, pd.Series("", index=df.index)).fillna("").astype(str)
            df[tf] = (t_series + " " + a_series).str.strip()
            print(f"[{self.tag}] rebuilt '{tf}' by merging '{title}' + '{abstract}'")
            return df

        # Neither fallback nor title/abstract are present — emit empty strings
        df[tf] = ""
        print(f"[{self.tag}] WARNING: no '{fallback}', '{title}', or '{abstract}' columns; "
              f"rebuilt '{tf}' as empty strings")
        return df


    # ------------------------------------------------------ run

    def run(self, bundle: DataBundle) -> None:
        # 1) Load inputs
        df: pd.DataFrame = self.load_path(bundle[self.needs[0]])
        terms_path: Path = bundle[self.needs[1]]
        terms_md = Path(terms_path).read_text(encoding="utf-8")

        print(f"\n[{self.tag}] ====================================================")
        print(f"[{self.tag}] df.shape          = {df.shape}")
        print(f"[{self.tag}] terms_md          = {terms_path}")

        if self.id_field not in df.columns:
            raise ValueError(f"DataFrame must contain an '{self.id_field}' column.")

        # 2) Hop split (optional)
        use_hops = self.init_settings.get("use_hops", False)
        if use_hops and "type" in df.columns:
            max_t   = df.type.max()
            df_prev = df[df.type < max_t]
            df_curr = df[df.type == max_t].reset_index(drop=True)
            print(f"[{self.tag}] hop split        → prev {df_prev.shape}, curr {df_curr.shape}")
        else:
            df_prev = df.iloc[0:0]
            df_curr = df.reset_index(drop=True)
            print(f"[{self.tag}] no hop split     → curr {df_curr.shape}")

        # 3) Ensure text column
        # df_curr = self._ensure_text_column(df_curr)
        df_curr = self._rebuild_text_column(df_curr)

        # 4) Output dir
        root = Path(bundle.get(SAVE_DIR_BUNDLE_KEY, "."))
        out  = root / self.tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{self.tag}] output dir        = {out}")

        # 5) Build Ocelot engine from Markdown
        oc = Ocelot.from_markdown(
            terms_md,
            positives_mode=self.call_settings["positives_mode"],
            global_positives_mode=self.call_settings["global_positives_mode"],
        )

        # 6) Prepare rows for Ocelot CSV-style API
        rows: List[Dict[str, str]] = [
            {
                self.id_field: str(r[self.id_field]),
                self.text_field: str(r[self.text_field]),
            }
            for _, r in df_curr[[self.id_field, self.text_field]].fillna("").astype(str).iterrows()
        ]
        print(f"[{self.tag}] rows prepared     = {len(rows)}  "
              f"(id_field='{self.id_field}', text_field='{self.text_field}')")

        # 7) Run filter inside Ocelot
        results = oc.filter_csv_rows(
            rows,
            text_field=self.text_field,
            id_field=self.id_field,
            emit_nonmatches=self.call_settings["emit_nonmatches"],
        )

        # 8) Build explain table
        table_cols = ["index", "passed", "matched_main", "matched_positives", "matched_negatives", "text"]
        ocelot_table = pd.DataFrame(results, columns=table_cols) if results else pd.DataFrame(columns=table_cols)
        print(f"[{self.tag}] explain rows      = {len(ocelot_table)}")

        # 9) Keep only passes for the merged df
        passed_ids = set(ocelot_table.loc[ocelot_table["passed"] == "true", "index"])
        kept = df_curr[df_curr[self.id_field].astype(str).isin(passed_ids)].copy()
        print(f"[{self.tag}] kept current      = {kept.shape}")

        # 10) Merge with prev hops and dedupe by id_field
        merged = (
            pd.concat([df_prev, kept], ignore_index=True)
            .drop_duplicates(subset=[self.id_field], keep="last")
            .reset_index(drop=True)
        )
        print(f"[{self.tag}] merge result      = {merged.shape}")

        # 11) Save artifacts + register
        df_path = out / f"{self.tag}.csv"
        table_path = out / "ocelot_table.csv"
        terms_out = out / "terms.md"

        merged.to_csv(df_path, index=False, encoding="utf-8-sig")
        ocelot_table.to_csv(table_path, index=False, encoding="utf-8-sig")
        Path(terms_out).write_text(terms_md, encoding="utf-8")

        self.register_checkpoint(self.provides[0], df_path)
        self.register_checkpoint(self.provides[1], table_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = merged
        bundle[f"{self.tag}.{self.provides[1]}"] = ocelot_table

        print(f"[{self.tag}] saved df          → {df_path}")
        print(f"[{self.tag}] saved table       → {table_path}")
        print(f"[{self.tag}] saved terms       → {terms_out}")
