from pathlib import Path
from typing import Dict, Sequence, Any, Tuple
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...post_processing import ArcticFox
from ...factorization import HNMFk

class ArticFoxBlock(AnimalBlock):

    CANONICAL_NEEDS = ("df", 'vocabulary', "model_path",)

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
            'clean_cols_name': self.col,
            'embedding_model': "SCINCL", 
        }
        default_call = {
            'ollama_model': "llama3.2:3b-instruct-fp16",  # Language model used for semantic label generation
            'label_clusters': True,             # Enable automatic labeling of clusters
            'generate_stats': True,             # Generate cluster-level statistics
            'process_parents': True,            # Propagate labels or stats upward through the hierarchy
            'skip_completed': True,             # Skip processing of nodes already labeled/stored
            'label_criteria': {                 # Rules to filter generated labels
                "minimum words": 2,
                "maximum words": 6
            },
            'label_info': {                     # Additional metadata to associate with generated labels
                 "source": "Science"
            },
            'number_of_labels': 5               # Number of candidate labels to generate per node
        }

        super().__init__(
            needs = needs,
            provides = provides,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            tag=tag,
            **kw,
        )


    def run(self, bundle: DataBundle) -> None:
        df = self.load_path(bundle[self.needs[0]])
        vocabulary = self.load_path(bundle[self.needs[1]])

        raw_model_path = str(bundle[self.needs[2]])
        # Try to resolve to an absolute path for traceability; fall back to the raw string.
        try:
            resolved_model_path = str(Path(raw_model_path).expanduser().resolve())
        except Exception:
            resolved_model_path = raw_model_path

        model = HNMFk(experiment_name=raw_model_path)
        model.load_model()

        pipeline = ArcticFox(model=model, **self.init_settings)
        # pipeline.run_full_pipeline(data_df=df, vocab=vocabulary, **self.call_settings)

        status_value = "Done"

        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(parents=True, exist_ok=True)
            status_file = out_dir / "status.txt"
            # Include model path info in the checkpointed status file
            status_file.write_text(
                f"status: {status_value}\n"
                f"model_path: {raw_model_path}\n"
                f"resolved_model_path: {resolved_model_path}\n",
                encoding="utf-8",
            )
            self.register_checkpoint(self.provides[0], status_file)

        bundle[f"{self.tag}.{self.provides[0]}"] = status_value



        