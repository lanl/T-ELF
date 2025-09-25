# pipeline/blocks/peacock_stats_block.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Literal
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY, RESULTS_DEFAULT

from .block_helpers.peacock_renderer import PeacockRenderer
from .block_helpers.hnmfk_paths import NodeSource, HNMFkNodeSource, GlobNodeSource


Mode = Literal["single", "hnmfk", "glob"]


class PeacockStatsBlock(AnimalBlock):
    CANONICAL_NEEDS: Tuple[str, ...] = ("df", )

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("outpath",),
        mode: Mode = "single",
        # renderer args
        hist_stats: Sequence[str] = ("paper_count", "num_citations"),
        hist_ylabels: Optional[Dict[str, str]] = None,
        col_names: Optional[Dict[str, str]] = None,
        affiliation_palette: Optional[Dict[str, str]] = None,
        country: Optional[str] = None,
        # discovery args
        experiment_path: Optional[str] = None,   # for mode="hnmfk"
        glob_root: Optional[str] = None,         # for mode="glob"
        skip_completed: bool = True,
        **kw: Any,
    ) -> None:
        self.mode = mode
        self.experiment_path = Path(experiment_path).expanduser().resolve() if experiment_path else None
        self.glob_root = Path(glob_root).expanduser().resolve() if glob_root else None
        self.skip_completed = skip_completed

        self.renderer = PeacockRenderer(
            hist_stats=hist_stats,
            hist_ylabels=hist_ylabels,
            col_names=col_names,
            affiliation_palette=affiliation_palette,
            country=country,
        )

        # require model_path if mode==hnmfk (via conditional_needs), but keep simple API
        conds = list(kw.pop("conditional_needs", ()))
        if self.mode == "hnmfk" and not self.experiment_path:
            conds.append(("model_path", lambda _b, _s: True))

        super().__init__(
            needs=needs,
            provides=provides,
            tag="PeacockStats",
            init_settings={},
            call_settings={},
            conditional_needs=tuple(conds),
            **kw,
        )

    # ——————————————————————————————————————————
    def _source(self, bundle: DataBundle) -> Optional[NodeSource]:
        if self.mode == "hnmfk":
            exp = self.experiment_path or Path(str(bundle["model_path"]))
            return HNMFkNodeSource(exp)
        if self.mode == "glob":
            root = self.glob_root or Path(bundle.get(SAVE_DIR_BUNDLE_KEY, RESULTS_DEFAULT))
            return GlobNodeSource(root)
        return None  # single mode

    def _parent_out_dir(self, bundle: DataBundle) -> Path:
        # where the block-level checkpoint lands if needed
        base = Path(bundle.get(SAVE_DIR_BUNDLE_KEY, RESULTS_DEFAULT))
        return base / self.tag

    # ——————————————————————————————————————————
    def run(self, bundle: DataBundle) -> None:
        # mode: single (unchanged behavior)
        if self.mode == "single":
            df: pd.DataFrame = bundle["df"]
            out_dir = Path(bundle.get(SAVE_DIR_BUNDLE_KEY, RESULTS_DEFAULT))
            self.renderer.render(df, out_dir)
            ckpt_dir = self._parent_out_dir(bundle); ckpt_dir.mkdir(parents=True, exist_ok=True)
            marker = ckpt_dir / "none.csv"; marker.write_text("status\nok\n")
            self.register_checkpoint(self.provides[0], marker)
            bundle[f"{self.tag}.{self.provides[0]}"] = marker
            return

        # mode: per-node (hnmfk or glob)
        source = self._source(bundle)
        assert source is not None, "Invalid configuration: per-node mode requires a NodeSource."

        produced = []
        for node in source.iter_nodes():
            out_dir = node.dir / "peacock"
            if self.skip_completed and (out_dir / "PeacockStats.done").exists():
                produced.append(out_dir)
                continue

            if not node.csv.exists():
                # skip nodes that haven't been post-processed yet
                continue

            df_local = pd.read_csv(node.csv)
            self.renderer.render(df_local, out_dir)
            (out_dir / "PeacockStats.done").write_text("ok")
            produced.append(out_dir)

        # block-level checkpoint
        ckpt_dir = self._parent_out_dir(bundle); ckpt_dir.mkdir(parents=True, exist_ok=True)
        registry = ckpt_dir / "per_node_registry.txt"
        registry.write_text("\n".join(map(str, produced)))
        self.register_checkpoint(self.provides[0], registry)
        bundle[f"{self.tag}.{self.provides[0]}"] = registry
