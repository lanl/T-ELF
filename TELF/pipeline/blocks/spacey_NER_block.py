from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

class SpacyNERBlock(AnimalBlock):
    """
    spaCy NER over one or more text columns (default: ['title', 'abstract']).

    Adds ONE new column to the DataFrame:
      - `output_column` (default: 'ner_by_label'): a STRING of a dictionary that
        can be round-tripped with `ast.literal_eval`. The dict maps:
            { <NER_LABEL>: [unique entity strings for the row] }

    Artifacts:
      - <SAVE_DIR>/<tag>/<tag>.csv    (enriched DataFrame)

    Requirements:
      - spaCy with a model installed (default: 'en_core_web_lg').

    Parameters
    ----------
    needs : tuple
        Bundle keys to read. Default: ("df",).
    provides : tuple
        Bundle keys to write. Default: ("df",).
    text_columns : Optional[List[str]]
        Which DF columns to run NER on. Default: ["title", "abstract"].
    id_field : str
        (Unused for output but kept for compatibility.) Default: "eid".
    spacy_model : str
        spaCy model name to load. Default: "en_core_web_lg".
    batch_size : int
        spaCy pipe batch size. Default: 256.
    n_process : int
        Number of processes for spaCy pipe. Default: 1.
    drop_existing : bool
        If True, drop an existing output column before writing. Default: True.
    output_column : str
        Name of the new column to write. Default: "ner_by_label".
    tag : str
        Block tag and artifact folder name. Default: "spaceyNER".
    init_settings : Optional[Dict[str, Any]]
        Extra init settings; merged into defaults.
    """

    CANONICAL_NEEDS = ("df",)

    def __init__(
        self,
        *,
        needs=CANONICAL_NEEDS,
        provides=("df",),
        text_columns: Optional[List[str]] = None,
        id_field: str = "eid",
        spacy_model: str = "en_core_web_lg",
        batch_size: int = 256,
        n_process: int = 1,
        drop_existing: bool = True,
        output_column: str = "ner_by_label",
        tag: str = "spaceyNER",
        init_settings: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        self.id_field = id_field

        default_init = {
            "text_columns": text_columns or ["title", "abstract"],
            "spacy_model": spacy_model,
            "batch_size": int(batch_size),
            "n_process": int(n_process),
            "drop_existing": bool(drop_existing),
            "output_column": output_column,
            "verbose": True,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings={},  # not used here
            **kw,
        )

    # ------------------------------- helpers

    def _load_spacy(self):
        try:
            import spacy  # local import so module is optional until needed
        except Exception as e:
            raise RuntimeError(
                f"[{self.tag}] spaCy is not installed. Please install spaCy and a model."
            ) from e

        model = self.init_settings["spacy_model"]
        try:
            # Disable components we don't need for speed.
            nlp = spacy.load(model, disable=["tagger", "lemmatizer", "textcat"])
        except OSError as e:
            raise RuntimeError(
                f"[{self.tag}] spaCy model '{model}' is not installed.\n"
                f"Install it with: python -m spacy download {model}"
            ) from e
        return nlp

    # ------------------------------- run

    def run(self, bundle: DataBundle) -> None:
        # 1) Load input DF
        df: pd.DataFrame = self.load_path(bundle[self.needs[0]])
        print(f"\n[{self.tag}] ====================================================")
        print(f"[{self.tag}] df.shape          = {df.shape}")

        text_cols: List[str] = list(self.init_settings.get("text_columns", ["title", "abstract"]))
        present = [c for c in text_cols if c in df.columns]
        missing = [c for c in text_cols if c not in df.columns]
        if not present:
            raise ValueError(f"[{self.tag}] None of the requested text columns are present: {text_cols}")
        if missing:
            print(f"[{self.tag}] WARNING: missing text columns {missing}; will skip them")

        # 2) Output dir
        root = Path(bundle.get(SAVE_DIR_BUNDLE_KEY, "."))
        out = root / self.tag
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{self.tag}] output dir        = {out}")

        # 3) Load spaCy
        nlp = self._load_spacy()
        batch_size = int(self.init_settings.get("batch_size", 256))
        n_process = int(self.init_settings.get("n_process", 1))
        drop_existing = bool(self.init_settings.get("drop_existing", True))
        out_col = str(self.init_settings.get("output_column", "ner_by_label"))

        # 4) Prepare DF; optionally drop existing output column
        df_proc = df.copy()
        if drop_existing and out_col in df_proc.columns:
            df_proc = df_proc.drop(columns=[out_col])
            print(f"[{self.tag}] dropped existing column: {out_col}")

        # We'll aggregate per-row across ALL present text columns.
        # For each row, maintain a dict[label] -> list[str] (unique, preserve insertion order)
        num_rows = len(df_proc)
        agg_by_row: List[Dict[str, List[str]]] = [dict() for _ in range(num_rows)]

        # 5) Run NER column-by-column and merge results per row
        for col in present:
            print(f"[{self.tag}] NER on column     = {col}")
            texts = df_proc[col].fillna("").astype(str).tolist()

            for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
                if not doc.ents:
                    continue
                row_dict = agg_by_row[i]
                for ent in doc.ents:
                    label = ent.label_
                    text = ent.text
                    lst = row_dict.setdefault(label, [])
                    # preserve insertion order & uniqueness
                    if text not in lst:
                        lst.append(text)

        # 6) Serialize each row's dict to a string compatible with ast.literal_eval
        # Using JSON ensures safe, unambiguous formatting; ast.literal_eval accepts JSON literals.
        serialized = [json.dumps(d, ensure_ascii=False) for d in agg_by_row]

        df_proc[out_col] = serialized

        # 7) Save artifacts + register
        df_path = out / f"{self.tag}.csv"
        df_proc.to_csv(df_path, index=False, encoding="utf-8-sig")
        self.register_checkpoint(self.provides[0], df_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = df_proc
        print(f"[{self.tag}] saved df          → {df_path}")
        print(f"[{self.tag}] added column      = {out_col}")