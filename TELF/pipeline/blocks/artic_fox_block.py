from pathlib import Path
from typing import Dict, Any
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...post_processing import ArcticFox
from ...factorization import HNMFk


class ArticFoxBlock(AnimalBlock):
    """
    Block wrapper for the ArcticFox post-process/label/stats pipeline.
    Use call_settings['steps'] to run any subset: ["post"], ["label"], ["stats"],
    or combinations like ["post","label"], ["label","stats"], ["post","stats"], ["post","label","stats"].
    If steps is None, legacy behavior uses the label_clusters/generate_stats booleans.
    """

    CANONICAL_NEEDS = ("df", "vocabulary", "model_path")

    def __init__(
        self,
        col: str = "clean_title_abstract",
        needs = CANONICAL_NEEDS,
        provides = ("block_status",),
        tag: str = "ArticFox",
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw,
    ) -> None:

        self.col = col
        default_init = {
            "clean_cols_name": self.col,
            "embedding_model": "SCINCL",
        }
        default_call = {
            "ollama_model": "llama3.2:3b-instruct-fp16",  # Language model used for semantic label generation
            "label_clusters": True,            # Back-compat: used when steps is None
            "generate_stats": True,            # Back-compat: used when steps is None
            "process_parents": True,
            "skip_completed": True,
            "label_criteria": {"minimum words": 2, "maximum words": 6},
            "label_info": {"source": "Science"},
            "number_of_labels": 5,
            # NEW: choose subset explicitly; None keeps legacy boolean behavior
            # Examples: ["post"], ["label"], ["stats"], ["post","label"], ["label","stats"], ["post","stats"], ["post","label","stats"]
            "steps": None,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            tag=tag,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        # Resolve inputs
        df = self.load_path(bundle[self.needs[0]])
        vocabulary = self.load_path(bundle[self.needs[1]])
        raw_model_path = str(bundle[self.needs[2]])

        try:
            resolved_model_path = str(Path(raw_model_path).expanduser().resolve())
        except Exception:
            resolved_model_path = raw_model_path

        # Load HNMFk model
        model = HNMFk(experiment_name=raw_model_path)
        model.load_model()

        # Run selected steps (order enforced inside ArcticFox)
        pipeline = ArcticFox(model=model, **self.init_settings)
        pipeline.run_full_pipeline(
            data_df=df,
            vocab=vocabulary,
            **self.call_settings
        )

        # Write a lightweight status checkpoint
        status_value = "Done"
        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(parents=True, exist_ok=True)
            status_file = out_dir / "status.txt"
            status_file.write_text(
                f"status: {status_value}\n"
                f"model_path: {raw_model_path}\n"
                f"resolved_model_path: {resolved_model_path}\n",
                encoding="utf-8",
            )
            self.register_checkpoint(self.provides[0], status_file)

        bundle[f"{self.tag}.{self.provides[0]}"] = status_value
