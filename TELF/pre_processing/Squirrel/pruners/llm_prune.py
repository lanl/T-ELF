import json
import logging
import math
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm.auto import tqdm

from ....helpers.llm import get_ollama_llm, vote_once, build_json_vote_prompt

log = logging.getLogger(__name__)

class LLMPruner:

    def __init__(
        self,
        llm_model_name: str,
        llm_api_url: str,
        llm_vote_trials: int,
        llm_promote_threshold: float,
        llm_temperature: float,
        verbose: bool = True,
    ):
        """
        Perform LLM-based refinement on an embedding-pruned dataset, annotating each
        document with an `llm_accept` boolean column.

        Parameters
        ----------
        llm_model_name : str
            Ollama model identifier (e.g. `"llama3.1:405b"`).
        llm_api_url : str
            Base URL for the Ollama API.
        llm_vote_trials : int
            Number of independent votes per document.
        llm_promote_threshold : float
            Fraction of “yes” votes required to accept a previously rejected doc.
        llm_temperature : float
            Sampling temperature for the LLM.
        verbose : bool
            Whether to show tqdm progress bars.
        """
        self.llm_vote_trials = llm_vote_trials
        self.llm_promote_threshold = llm_promote_threshold
        self.verbose = verbose
        self.NAME = 'llm_prune'

        self.llm = get_ollama_llm(
            model=llm_model_name,
            base_url=llm_api_url,
            temperature=llm_temperature
        )

    def _vote_on_document(self, df, idx, row, ctx, keep, fout, data_column) -> None:
        """
        Perform the LLM voting process on a single document.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        idx : int
            Index of the current row in the DataFrame.
        row : pd.Series
            The current row in the DataFrame.
        ctx : list
            Context for LLM voting.
        keep : set
            Set of indices to keep (i.e., already accepted).
        fout : file
            File handler to write the voting record.
        data_column : str
            Column name containing the data to be voted on.
        """
        prompt = build_json_vote_prompt(row[data_column], ctx)
        votes = [vote_once(self.llm, prompt) for _ in range(self.llm_vote_trials)]
        yes_count = sum(int(v[0]) for v in votes)
        reasons = [v[1] for v in votes]

        threshold =  math.ceil(self.llm_promote_threshold * self.llm_vote_trials)
        decision = yes_count >= threshold
        df.at[idx, self.NAME] = decision

        record = {
            "index": idx,
            "answers": ["yes" if v[0] else "no" for v in votes],
            'reasons': reasons,
            "final": decision,
        }
        fout.write(json.dumps(record) + "\n")

        if decision:
            keep.add(idx)

    def __call__(self, df, output_dir, label_column, reference_label, data_column) -> pd.DataFrame:
        """
        Run LLM voting across all rows, annotate `llm_accept`, save vote
        records and the annotated DataFrame, and return it.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be processed.
        output_dir : str or Path
            Directory to save the output files.
        label_column : str
            Column name indicating class labels.
        reference_label : Union[int, str]
            Value in `label_column` used to build context examples.
        data_column : str
            Column name containing the data to be voted on.

        Returns
        -------
        pd.DataFrame
            The input DataFrame augmented with `llm_accept` .
        """

        output_dir = Path(output_dir) / "llm_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize required paths and context for LLM voting
        decisions_path = output_dir / "llm_decisions.jsonl"
        keep = set(df[(df[label_column] == reference_label)].index)

        # Open the decisions file to save the vote records
        with decisions_path.open("a", encoding="utf-8") as fout:
            for idx, row in tqdm(df.iterrows(), total=len(df), disable=not self.verbose, desc="LLM voting"):
                if idx in keep:
                    df.at[idx, self.NAME] = True
                    continue

                # Perform voting and record results
                ctx = df.head(10)[data_column].tolist()
                self._vote_on_document(df, idx, row, ctx, keep, fout, data_column)

        # Save the annotated DataFrame
        out_path = output_dir / "llm_pruned.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved LLM-annotated DataFrame → %s", out_path)

        return df
