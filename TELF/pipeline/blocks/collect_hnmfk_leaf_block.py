# TELF/pipeline/blocks/leaf_data_labels_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional, List
import os
import re
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class CollectHNMFkLeafBlock(AnimalBlock):
    """
    Build two artifacts from a completed HNMFk run:

      • LEAF_DATA.csv
          ALL columns from the input df for documents in each LEAF node,
          plus:  cluster (int), Graph_Name (e.g. 'depth_2_Parent_1_0')

      • LEAF_LABELS.csv
          Columns: Graph_Name, label, words
          (words are comma-joined from top_words.csv for each cluster)

      • summary.txt
          Total documents, number of leaf clusters, and per-cluster counts.

    Resolution order for HNMFk experiment directory:
      1) From bundle key (default: 'SemanticHNMFk.model_path', overridable)
      2) call_settings['hnmfk_dir']
      3) Auto-discover under SAVE_DIR_BUNDLE_KEY using tag (default: 'SemanticHNMFk')

    Optional call_settings:
      - hnmfk_dir: explicit path to the experiment directory
      - hnmfk_tag: tag name to look for (default: 'SemanticHNMFk')
      - hnmfk_bundle_key: bundle key for model path (default: f'{hnmfk_tag}.model_path')

    Provides:
      - 'leaf_data_csv'    → Path to LEAF_DATA.csv
      - 'leaf_labels_csv'  → Path to LEAF_LABELS.csv
      (summary.txt is written alongside these outputs)
    """

    CANONICAL_NEEDS: Tuple[str, ...] = ("df", SAVE_DIR_BUNDLE_KEY)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("leaf_data_csv", "leaf_labels_csv"),
        tag: str = "LeafDataLabels",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=init_settings or {},
            call_settings=call_settings or {},
            verbose=verbose,
            checkpoint=True,
            load_checkpoint=False,  # force a real run the first time
            **kw,
        )

    # ───────────────────────── helpers ─────────────────────────

    @staticmethod
    def _to_list(x, *, as_int=False, as_str=False) -> List[Any]:
        if x is None:
            out = []
        elif isinstance(x, list):
            out = x
        elif isinstance(x, (tuple, set)):
            out = list(x)
        elif hasattr(x, "tolist"):
            out = x.tolist()
        else:
            try:
                out = list(x)
            except TypeError:
                out = [x]
        if as_int:
            res = []
            for v in out:
                try:
                    res.append(int(v))
                except Exception:
                    pass
            return res
        if as_str:
            return [str(v) for v in out]
        return out

    @staticmethod
    def _find_semantic_dir(root: Path, tag_name: str = "SemanticHNMFk") -> Optional[Path]:
        root = root.resolve()
        candidates = [p for p in root.glob(f"*_{tag_name}") if p.is_dir()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0].resolve()
        plain = (root / tag_name)
        return plain.resolve() if plain.is_dir() else None

    @staticmethod
    def _load_pickle(path: Path):
        with path.open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _load_checkpoint(exp_dir: Path) -> Dict[str, Any]:
        """Try common checkpoint names; if not found, search recursively for 'checkpoint*'."""
        exp_dir = exp_dir.resolve()
        # Primary expected name
        ckpt = (exp_dir / "checkpoint.p").resolve()
        candidates: List[Path] = []
        if ckpt.is_file():
            candidates = [ckpt]
        else:
            alternates = [
                exp_dir / "checkpoint.pkl",
                exp_dir / "checkpoint.pickle",
                exp_dir / "checkpoint",
            ]
            candidates = [c.resolve() for c in alternates if c.is_file()]
            if not candidates:
                globbed = [g.resolve() for g in exp_dir.glob("checkpoint*") if g.is_file()]
                if globbed:
                    candidates = globbed
            if not candidates:
                # NEW: recursive search fallback
                deep = [g.resolve() for g in exp_dir.rglob("checkpoint*") if g.is_file()]
                if deep:
                    # newest first
                    deep.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    candidates = [deep[0]]

        if not candidates:
            return {}

        ckpt = candidates[0]
        try:
            # FIX: correct class reference
            return CollectHNMFkLeafBlock._load_pickle(ckpt)
        except Exception:
            return {}

    @staticmethod
    def _safe_rebase(path: str, old_base: Optional[str], new_base: Optional[str]) -> str:
        if not path:
            return path
        p_norm = os.path.normpath(path)
        if not old_base or not new_base:
            return str(Path(p_norm).resolve())
        try:
            old_base_n = os.path.normpath(old_base)
            new_base_n = os.path.normpath(new_base)
            if p_norm.startswith(old_base_n):
                rel = os.path.relpath(p_norm, old_base_n)
                return str(Path(os.path.join(new_base_n, rel)).resolve())
        except Exception:
            pass
        try:
            old_seg = re.search(r"(\d+_SemanticHNMFk|SemanticHNMFk)", str(old_base)).group(1)
            new_seg = re.search(r"(\d+_SemanticHNMFk|SemanticHNMFk)", str(new_base)).group(1)
            return str(Path(p_norm.replace(old_seg, new_seg)).resolve())
        except Exception:
            return str(Path(p_norm).resolve())

    def _log(self, fp: Path, *msgs: str) -> None:
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.parent.mkdir(parents=True, exist_ok=True)
            with fp.open("a", encoding="utf-8") as f:
                for m in msgs:
                    f.write(f"[{ts}] {m}\n")
        except Exception:
            pass

    # ─────────────────────────── run ────────────────────────────

    def run(self, bundle: DataBundle) -> None:
        df: pd.DataFrame = bundle["df"]
        root_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]).expanduser().resolve()

        out_dir = (root_dir / self.tag).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        log_fp = out_dir / "debug.log"

        # Output paths (exact filenames requested)
        leaf_data_csv   = out_dir / "LEAF_DATA.csv"
        leaf_labels_csv = out_dir / "LEAF_LABELS.csv"
        summary_txt     = out_dir / "summary.txt"  # NEW

        # Resolve HNMFk experiment dir — ORDER: bundle → call_settings → auto
        tag_name = self.call_settings.get("hnmfk_tag", "SemanticHNMFk")
        bundle_key = self.call_settings.get("hnmfk_bundle_key", f"{tag_name}.model_path")

        exp_dir: Optional[Path] = None
        src = ""

        # 1) from bundle (model_path preferred, else model object)
        if bundle_key in bundle:
            exp_dir = Path(bundle[bundle_key]).expanduser().resolve()
            src = f"bundle[{bundle_key}]"
        elif f"{tag_name}.model" in bundle:
            try:
                model_obj = bundle[f"{tag_name}.model"]
                exp_dir = Path(getattr(model_obj, "experiment_save_path")).expanduser().resolve()
                src = f"bundle[{tag_name}.model]"
            except Exception:
                exp_dir = None

        # 2) explicit in call_settings
        if exp_dir is None and self.call_settings.get("hnmfk_dir"):
            exp_dir = Path(self.call_settings["hnmfk_dir"]).expanduser().resolve()
            src = "call_settings.hnmfk_dir"

        # 3) auto-discover next to SAVE_DIR
        if exp_dir is None:
            exp_dir = self._find_semantic_dir(root_dir, tag_name=tag_name)
            src = f"auto({root_dir})"

        if not exp_dir or not exp_dir.exists():
            self._log(log_fp, f"HNMFk dir not found (src={src}). Writing empty outputs.")
            pd.DataFrame(columns=list(df.columns) + ["cluster", "Graph_Name"]).to_csv(leaf_data_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(columns=["Graph_Name", "label", "words"]).to_csv(leaf_labels_csv, index=False, encoding="utf-8-sig")
            # NEW: still write an empty summary.txt
            with summary_txt.open("w", encoding="utf-8") as f:
                f.write("Total documents: 0\nLeaf clusters: 0\n")
            self.register_checkpoint("leaf_data_csv", leaf_data_csv)
            self.register_checkpoint("leaf_labels_csv", leaf_labels_csv)
            bundle[f"{self.tag}.leaf_data_csv"] = leaf_data_csv
            bundle[f"{self.tag}.leaf_labels_csv"] = leaf_labels_csv
            if self.verbose:
                print(f"[{self.tag}] Wrote:\n  {leaf_data_csv}\n  {leaf_labels_csv}\n  {summary_txt}")
            return

        ckpt = self._load_checkpoint(exp_dir)
        if not ckpt:
            self._log(log_fp, f"No checkpoint file in {exp_dir}. Writing empty outputs.")
            pd.DataFrame(columns=list(df.columns) + ["cluster", "Graph_Name"]).to_csv(leaf_data_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(columns=["Graph_Name", "label", "words"]).to_csv(leaf_labels_csv, index=False, encoding="utf-8-sig")
            with summary_txt.open("w", encoding="utf-8") as f:
                f.write("Total documents: 0\nLeaf clusters: 0\n")
            self.register_checkpoint("leaf_data_csv", leaf_data_csv)
            self.register_checkpoint("leaf_labels_csv", leaf_labels_csv)
            bundle[f"{self.tag}.leaf_data_csv"] = leaf_data_csv
            bundle[f"{self.tag}.leaf_labels_csv"] = leaf_labels_csv
            if self.verbose:
                print(f"[{self.tag}] Wrote:\n  {leaf_data_csv}\n  {leaf_labels_csv}\n  {summary_txt}")
            return

        node_save_paths: Dict[str, str] = ckpt.get("node_save_paths", {}) or {}
        root_name: str = ckpt.get("root_name") or "Root"
        old_base = ckpt.get("experiment_name") or ckpt.get("experiment_save_path")
        new_base = str(exp_dir)

        final_df_parts: List[pd.DataFrame] = []
        label_parts: List[pd.DataFrame] = []
        visited = 0
        leaves_seen = 0

        def _resolve(p: str) -> str:
            return self._safe_rebase(p, old_base, new_base)

        def _load_node_safely(path: Path):
            try:
                return self._load_pickle(path)
            except Exception as e:
                self._log(log_fp, f"Failed to load node: {path} :: {e}")
                return None

        def visit(node_name: str):
            nonlocal visited, leaves_seen
            if node_name not in node_save_paths:
                self._log(log_fp, f"node_save_paths missing entry for '{node_name}'")
                return
            node_path = Path(_resolve(node_save_paths[node_name])).expanduser().resolve()
            visited += 1
            if not node_path.exists():
                self._log(log_fp, f"node pickle missing: {node_path}")
                return

            node_obj = _load_node_safely(node_path)
            if node_obj is None:
                return

            # children
            for cname in self._to_list(getattr(node_obj, "child_node_names", []), as_str=True):
                visit(cname)

            # leaf?
            if not bool(getattr(node_obj, "leaf", False)):
                return

            leaves_seen += 1
            original_indices = self._to_list(getattr(node_obj, "original_indices", None), as_int=True)
            if not original_indices:
                self._log(log_fp, f"leaf with empty original_indices: {node_path}")
                return

            # extract ALL columns for those rows
            try:
                node_df = df.iloc[original_indices].copy()
            except Exception as e:
                self._log(log_fp, f"iloc failed for indices (len={len(original_indices)}): {e}")
                return

            # num_clusters from W or signature
            W = getattr(node_obj, "W", None)
            if W is None:
                sig = getattr(node_obj, "signature", None)
                if sig is None:
                    num_clusters = 1
                else:
                    sig = np.asarray(sig)
                    if sig.ndim == 1:
                        sig = sig.reshape(-1, 1)
                    num_clusters = int(sig.shape[1])
            else:
                W = np.asarray(W)
                if W.ndim == 1:
                    W = W.reshape(-1, 1)
                num_clusters = int(W.shape[1])

            node_dir = node_path.parent
            cluster_csv = node_dir / f"cluster_for_k={num_clusters}.csv"
            cluster_membership = None
            if cluster_csv.is_file():
                try:
                    cluster_membership = pd.read_csv(cluster_csv)
                except Exception as e:
                    self._log(log_fp, f"read_csv failed: {cluster_csv} :: {e}")

            # attach cluster column
            if isinstance(cluster_membership, pd.DataFrame) and "cluster" in cluster_membership.columns:
                if len(cluster_membership) == len(node_df):
                    node_df["cluster"] = cluster_membership["cluster"].to_numpy()
                else:
                    # try to align if cluster CSV has a 'doc_index' column
                    if "doc_index" in cluster_membership.columns:
                        try:
                            aligned = cluster_membership.set_index("doc_index").loc[original_indices]
                            node_df["cluster"] = aligned["cluster"].to_numpy()
                        except Exception:
                            m = min(len(cluster_membership), len(node_df))
                            node_df = node_df.iloc[:m].copy()
                            node_df["cluster"] = cluster_membership["cluster"].iloc[:m].to_numpy()
                            self._log(log_fp, f"cluster length mismatch; truncated to {m} :: {cluster_csv}")
                    else:
                        m = min(len(cluster_membership), len(node_df))
                        node_df = node_df.iloc[:m].copy()
                        node_df["cluster"] = cluster_membership["cluster"].iloc[:m].to_numpy()
                        self._log(log_fp, f"cluster length mismatch; truncated to {m} :: {cluster_csv}")
            else:
                node_df["cluster"] = 0  # default single cluster

            # Graph_Name from parent dir of node pickle + cluster id
            graph_name_part = os.path.basename(os.path.dirname(str(node_path)))
            for k_val in sorted(pd.unique(node_df["cluster"])):
                sub = node_df[node_df["cluster"] == k_val].copy()
                sub["Graph_Name"] = f"{graph_name_part}_{k_val}"
                final_df_parts.append(sub)

            # labels
            cs_fp = node_dir / "cluster_summaries.csv"
            tw_fp = node_dir / "top_words.csv"
            if cs_fp.is_file() and tw_fp.is_file():
                try:
                    cs = pd.read_csv(cs_fp)
                    tw = pd.read_csv(tw_fp)
                    words_map: Dict[str, str] = {}
                    for col in tw.columns:
                        words_map[str(col)] = ",".join(tw[col].dropna().astype(str).tolist())
                    cs = cs.copy()
                    cs["cluster"] = cs["cluster"].astype(str)
                    cs["words"] = cs["cluster"].map(words_map).fillna("")
                    cs["Graph_Name"] = cs["cluster"].apply(lambda c: f"{graph_name_part}_{c}")
                    label_parts.append(cs[["Graph_Name", "label", "words"]])
                except Exception as e:
                    self._log(log_fp, f"label build failed at {node_dir}: {e}")

        # Walk from root
        if root_name not in node_save_paths:
            self._log(log_fp, f"root_name '{root_name}' missing in node_save_paths; nothing to traverse.")
        else:
            visit(root_name)

        # Concatenate & write (always write something)
        if final_df_parts:
            final_df = pd.concat(final_df_parts, ignore_index=True)
        else:
            final_df = pd.DataFrame(columns=list(df.columns) + ["cluster", "Graph_Name"])
        final_df.to_csv(leaf_data_csv, index=False, encoding="utf-8-sig")

        if label_parts:
            labels_df = pd.concat(label_parts, ignore_index=True)
        else:
            labels_df = pd.DataFrame(columns=["Graph_Name", "label", "words"])
        labels_df.to_csv(leaf_labels_csv, index=False, encoding="utf-8-sig")

        # NEW: Write summary.txt (totals + per-cluster counts)
        try:
            if not final_df.empty and "Graph_Name" in final_df.columns:
                counts = final_df.groupby("Graph_Name").size().sort_values(ascending=False)
            else:
                counts = pd.Series(dtype=int)
            total_docs = int(len(final_df))
            num_leaf_clusters = int(len(counts))
            lines = [
                f"Total documents: {total_docs}",
                f"Leaf clusters: {num_leaf_clusters}",
                "",
            ]
            lines += [f"{name}\t{int(cnt)}" for name, cnt in counts.items()]
            with summary_txt.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            self._log(log_fp, f"Failed to write summary.txt: {e}")

        # Checkpoint + bundle exposure
        self.register_checkpoint("leaf_data_csv", leaf_data_csv)
        self.register_checkpoint("leaf_labels_csv", leaf_labels_csv)
        bundle[f"{self.tag}.leaf_data_csv"] = leaf_data_csv
        bundle[f"{self.tag}.leaf_labels_csv"] = leaf_labels_csv
        # Optional: also expose summary path in the bundle (no need to register)
        bundle[f"{self.tag}.summary_txt"] = summary_txt  # NEW

        # Summary in logs + stdout (if verbose)
        self._log(
            log_fp,
            f"exp_dir={exp_dir}",
            f"nodes_seen={visited}, leaves_seen={leaves_seen}",
            f"wrote LEAF_DATA rows={len(final_df)}",
            f"wrote LEAF_LABELS rows={len(labels_df)}",
            f"wrote summary at {summary_txt}",
        )
        if self.verbose:
            print(f"[{self.tag}] Wrote:\n  {leaf_data_csv}\n  {leaf_labels_csv}\n  {summary_txt}")
