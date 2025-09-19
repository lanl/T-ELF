from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

class SpacyNERBlock(AnimalBlock):
    """
    spaCy NER over one or more text columns (default: ['title', 'abstract']).

    For each specified text column `col`, adds:
      - f"{col}_ents": JSON list of dicts {text, label, start, end}
      - f"{col}_ents_by_label": JSON dict of {label: [unique entity strings]}

    Artifacts:
      - <SAVE_DIR>/spaceyNER/spaceyNER.csv             (enriched DataFrame)
      - <SAVE_DIR>/spaceyNER/entities.csv              (optional, exploded entity rows)

    Requirements:
      - spaCy with a model installed (default: 'en_core_web_sm').

    Parameters
    ----------
    needs : tuple
        Bundle keys to read. Default: ("df",).
    provides : tuple
        Bundle keys to write. Default: ("df", "ents_table").
        If you pass only ("df",), the exploded entities table is skipped.
    text_columns : Optional[List[str]]
        Which DF columns to run NER on. Default: ["title", "abstract"].
    id_field : str
        Row identifier, used in the exploded entities table. Default: "eid".
    spacy_model : str
        spaCy model name to load. Default: "en_core_web_sm".
    batch_size : int
        spaCy pipe batch size. Default: 256.
    n_process : int
        Number of processes for spaCy pipe. Default: 1.
    drop_existing : bool
        If True, drop and rebuild any existing NER output columns. Default: True.
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
        provides=("df", "ents_table"),
        text_columns: Optional[List[str]] = None,
        id_field: str = "eid",
        spacy_model: str = "en_core_web_lg",
        batch_size: int = 256,
        n_process: int = 1,
        drop_existing: bool = True,
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

        # 4) Prepare destination columns; optionally drop existing
        dest_cols = []
        for col in present:
            dest_cols += [f"{col}_ents", f"{col}_ents_by_label"]
        if drop_existing:
            to_drop = [c for c in dest_cols if c in df.columns]
            if to_drop:
                df = df.drop(columns=to_drop)
                print(f"[{self.tag}] dropped existing NER columns: {to_drop}")

        df_proc = df.copy()
        exploded_rows: List[Dict[str, Any]] = []

        # 5) Run NER column-by-column
        for col in present:
            print(f"[{self.tag}] NER on column     = {col}")
            texts = df_proc[col].fillna("").astype(str).tolist()
            ents_json_col: List[str] = []
            bylabel_json_col: List[str] = []

            i_to_row_id = (
                df_proc[self.id_field].tolist() if self.id_field in df_proc.columns else list(range(len(df_proc)))
            )

            for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
                ents = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": int(ent.start_char),
                        "end": int(ent.end_char),
                    }
                    for ent in doc.ents
                ]

                by_label: Dict[str, List[str]] = {}
                for ent in doc.ents:
                    lst = by_label.setdefault(ent.label_, [])
                    if ent.text not in lst:
                        lst.append(ent.text)

                ents_json_col.append(json.dumps(ents, ensure_ascii=False))
                bylabel_json_col.append(json.dumps(by_label, ensure_ascii=False))

                # Collect exploded rows
                rid = i_to_row_id[i]
                for ent in doc.ents:
                    exploded_rows.append(
                        {
                            self.id_field: rid,
                            "source_column": col,
                            "text": ent.text,
                            "label": ent.label_,
                            "start": int(ent.start_char),
                            "end": int(ent.end_char),
                        }
                    )

            df_proc[f"{col}_ents"] = ents_json_col
            df_proc[f"{col}_ents_by_label"] = bylabel_json_col

        # 6) Save artifacts + register
        df_path = out / f"{self.tag}.csv"
        df_proc.to_csv(df_path, index=False, encoding="utf-8-sig")
        self.register_checkpoint(self.provides[0], df_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = df_proc
        print(f"[{self.tag}] saved df          → {df_path}")

        # optional exploded entities table
        if len(self.provides) > 1:
            ents_df = pd.DataFrame(
                exploded_rows,
                columns=[self.id_field, "source_column", "text", "label", "start", "end"],
            )
            table_path = out / "entities.csv"
            ents_df.to_csv(table_path, index=False, encoding="utf-8-sig")
            self.register_checkpoint(self.provides[1], table_path)
            bundle[f"{self.tag}.{self.provides[1]}"] = ents_df
            print(f"[{self.tag}] saved entities    → {table_path}")
