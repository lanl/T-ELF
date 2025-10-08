# blocks/semantic_hnmfk_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as ss
import traceback
import warnings

from ...factorization import HNMFk
from ...pre_processing import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# ------------------------------------------------------------------ #
# Helper – callback to rebuild a doc-word matrix at every node       #
# ------------------------------------------------------------------ #
class CustomSemanticCallback:
    def __init__(
        self,
        df: pd.DataFrame,
        vocabulary: list,
        *,
        target_column: str = "clean_title_abstract",
        options: Optional[Dict[str, Any]] = None,
        matrix_type: str = "tfidf",
    ) -> None:
        self.df = df
        self.vocabulary = vocabulary
        self.target_column = target_column
        self.options = options or {"min_df": 5, "max_df": 0.5}
        self.options["vocabulary"] = self.vocabulary
        self.matrix_type = matrix_type

    def __call__(self, original_idx: np.ndarray):
        sub_df = (
            self.df.iloc[original_idx]
            .copy(deep=True)
            .reset_index(drop=True)
        )

        # HARD RESET of Arrow-backed strings
        sub_df.columns = pd.Index([str(c) for c in sub_df.columns], dtype="object")
        if "string" in str(sub_df[self.target_column].dtype):
            sub_df[self.target_column] = sub_df[self.target_column].astype("object")

        beaver = Beaver()
        try:
            X, vocab = beaver.documents_words(
                dataset=sub_df,
                target_column=self.target_column,
                options=self.options,
                matrix_type=self.matrix_type,
                save_path=None,
            )
            return X.T.tocsr(), {"vocab": vocab}
        # except Exception:
        #     return ss.csr_matrix([[1]]), {
        #         "stop_reason": "documents_words could not build matrix",
        #     }
        except Exception as e:
            tb = traceback.format_exc()
            warnings.warn(f"documents_words failed on {len(original_idx)} docs: {e}\n{tb}")

            # Preserve the docs dimension so you can see it's not 1
            n_docs = int(len(original_idx))
            placeholder = ss.csr_matrix((1, n_docs))  # 1 feature × N docs for H-mode
            return placeholder, {
                "stop_reason": "documents_words failed",
                "exception": repr(e),
                "traceback": tb,
                "n_docs": n_docs,
            }

# ------------------------------------------------------------------ #
# Main block                                                        #
# ------------------------------------------------------------------ #
class SemanticHNMFkBlock(AnimalBlock):
    CANONICAL_NEEDS = ("X", "df", "vocabulary")

    def __init__(
        self,
        *,
        col: str = "clean_title_abstract",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("model", "model_path"),
        checkpoint_keys=("model_path",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "SemanticHNMFk",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        self.col = col

        # ---- defaults --------------------------------------------------
        nmfk_params = {
            "n_perturbs": 2,
            "n_iters": 2,
            "epsilon": 0.015,
            "n_jobs": -1,
            "init": "nnsvd",
            "use_gpu": False,
            "save_output": True,
            "collect_output": True,
            "predict_k_method": "sill",
            "verbose": True,
            "nmf_verbose": False,
            "transpose": False,
            "sill_thresh": 0.8,
            "pruned": True,
            "nmf_method": "nmf_fro_mu",
            "calculate_error": True,
            "predict_k": True,
            "use_consensus_stopping": 0,
            "calculate_pac": True,
            "consensus_mat": True,
            "perturb_type": "uniform",
            "perturb_multiprocessing": False,
            "perturb_verbose": False,
            "simple_plot": True,
            "k_search_method": "bst_pre",
            "H_sill_thresh": 0.1,
            "clustering_method": "kmeans",
            "device": -1,
        }

        default_init = {
            "nmfk_params": [nmfk_params],
            "cluster_on": "H",
            "depth": 1,
            "sample_thresh": 5,
            "K2": False,
            "Ks_deep_min": 1,
            "Ks_deep_max": 20,
            "Ks_deep_step": 1,
            "random_identifiers": False,
            "root_node_name": "Root",
        }

        default_call = {
            "Ks": range(1, 21),
            "from_checkpoint": True,
            "save_checkpoint": True,
        }
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            verbose=verbose,
            checkpoint_keys=checkpoint_keys,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        # 1 ▸ load input matrix
        X_val = bundle[self.needs[0]]
        X = self.load_path(X_val) if isinstance(X_val, (str, Path)) else X_val

        # 2 ▸ assemble HNMFk kwargs
        init_cfg = dict(self.init_settings)
        if "experiment_name" not in init_cfg and SAVE_DIR_BUNDLE_KEY in bundle:
            init_cfg["experiment_name"] = (
                Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            )

        init_cfg["generate_X_callback"] = CustomSemanticCallback(
            df=bundle[self.needs[1]],
            vocabulary=bundle[self.needs[2]],
            target_column=self.col,
        )

        # 3 ▸ fit hierarchy & checkpoint
        model = HNMFk(**init_cfg)
        model.fit(X, **self.call_settings)

         # 4 ▸ persist only the path in the checkpoint and in the bundle
        model_dir = model.experiment_save_path
        self.register_checkpoint(self.provides[1], model_dir)
        bundle[f"{self.tag}.{self.provides[1]}"] = model_dir

        # 5 ▸ immediately re-load the model so 'model' is always provided
        reloaded = HNMFk(experiment_name=model_dir)
        reloaded.load_model()
        bundle[f"{self.tag}.{self.provides[0]}"] = reloaded

    def _after_checkpoint_skip(self, bundle: DataBundle) -> None:
        # when re‐running from an existing checkpoint, load into memory
        model_dir = bundle[f"{self.tag}.{self.provides[1]}"]
        model = HNMFk(experiment_name=model_dir)
        model.load_model()
        bundle[f"{self.tag}.{self.provides[0]}"] = model
