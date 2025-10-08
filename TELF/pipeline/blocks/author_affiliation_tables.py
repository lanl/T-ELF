# blocks/affiliations_and_authors_block.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any, Optional, List, Tuple, Union

import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# your helpers live in blocks/block_helpers/
from .block_helpers.affiliation_partition import generate_top_affiliations_with_country
from .block_helpers.author_partition import write_top_authors_by_cluster


class AffiliationsAndAuthorsBlock(AnimalBlock):
    """
    Compute (affiliation, country, year) paper counts and top authors by cluster.

    ─────────────────────────────────────────────────────────────
    always needs : ('df',)   – accepts a CSV path OR a pandas.DataFrame
    provides     : ('affiliations_df', 'affiliations_csv',
                    'authors_df',      'authors_csv')
    tag          : 'AffilsAndAuthors' (namespace for its outputs)

    Results are written under  <bundle[SAVE_DIR_BUNDLE_KEY]>/<tag>/ .
    Checkpoints persist the two CSV paths so the block can be skipped on re-run.
    """

    CANONICAL_NEEDS = ("df",)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("affiliations_df", "affiliations_csv",
                                   "authors_df",      "authors_csv"),
        # Persist only the CSVs; DataFrames are rebuilt on load if needed.
        checkpoint_keys: Sequence[str] = ("affiliations_csv", "authors_csv"),
        conditional_needs: Sequence[tuple[str, Any]] = (),
        tag: str = "AffilsAndAuthors",
        # Defaults mirror your helper signatures
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:

        default_init: Dict[str, Any] = {}

        default_call: Dict[str, Any] = {
            # generate_top_affiliations_with_country(...)
            "min_total_papers": 20,
            "country_filter": None,            # exact match (including 'unknown') or None
            "partition_by_year": False,
            "per_year_output_dir": None,       # if None and partition_by_year=True → use <tag>/by_year

            # write_top_authors_by_cluster(...)
            "countries": None,                 # list[str] or None
            "top_n": 10,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=list(conditional_needs or []),
            checkpoint_keys=checkpoint_keys,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

    # ─────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────
    def _ensure_input_csv(self, bundle: DataBundle) -> Path:
        """
        Accepts either a DataFrame or a path in bundle['df'].
        If a DataFrame, persist it to <save_dir>/<tag>/input.csv and return that path.
        """
        src = bundle[self.needs[0]]
        save_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(src, pd.DataFrame):
            inp = save_dir / "input.csv"
            src.to_csv(inp, index=False, encoding="utf-8-sig")
            return inp

        # let AnimalBlock’s path rewriter handle legacy numbered dirs
        return Path(src)

    # ─────────────────────────────────────────────────────────────
    # work
    # ─────────────────────────────────────────────────────────────
    def run(self, bundle: DataBundle) -> None:
        df_path = self._ensure_input_csv(bundle)

        out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # === 1) Affiliations with country (and per-year optional) ===
        affils_csv = out_dir / "affiliations_top.csv"
        per_year_dir = self.call_settings.get("per_year_output_dir")
        if self.call_settings.get("partition_by_year") and not per_year_dir:
            per_year_dir = out_dir / "by_year"

        generate_top_affiliations_with_country(
            df_path=df_path,
            affils_output_path=affils_csv,
            min_total_papers=int(self.call_settings["min_total_papers"]),
            country_filter=self.call_settings.get("country_filter"),
            partition_by_year=bool(self.call_settings.get("partition_by_year")),
            per_year_output_dir=per_year_dir,
        )

        # read back for the bundle
        aff_df = pd.read_csv(affils_csv) if affils_csv.is_file() else pd.DataFrame(
            columns=["affiliation_name", "country", "year", "paper_count"]
        )

        # register checkpoints / provide
        self.register_checkpoint("affiliations_csv", affils_csv)
        bundle[f"{self.tag}.affiliations_csv"] = str(affils_csv)
        bundle[f"{self.tag}.affiliations_df"] = aff_df

        # === 2) Top authors by cluster =============================
        authors_csv = out_dir / "top_authors_by_cluster.csv"
        # Note: helper param name is COUNTY_NAMES (kept as-is)
        result_df = write_top_authors_by_cluster(
            df_path=str(df_path),
            output_path=str(authors_csv),
            COUNTY_NAMES=self.call_settings.get("countries"),
            top_n=int(self.call_settings.get("top_n", 10)),
        )

        # register checkpoints / provide
        self.register_checkpoint("authors_csv", authors_csv)
        bundle[f"{self.tag}.authors_csv"] = str(authors_csv)
        bundle[f"{self.tag}.authors_df"] = result_df