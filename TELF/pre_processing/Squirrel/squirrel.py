import pandas as pd
import logging
from pathlib import Path
from typing import Union, List
from .pruners.embed_prune import EmbeddingPruner

log = logging.getLogger(__name__)

class Squirrel:
    """
    Orchestrate a sequence of pruners that each take a DataFrame and return a DataFrame.
    """
    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        output_dir: Union[str, Path],
        pipeline: List,
        label_column = 'type',
        reference_label = 0,
        aggregrate_prune = True,
        data_column = 'title_abstract',
    ):
        """
        Parameters
        ----------
        data_source : str | Path | pd.DataFrame
            CSV path or initial DataFrame to process.
        output_dir : str | Path
            Base directory for pruner outputs.
        pipeline : list
            List of pruner instances; each __call__ must return a DataFrame.
        """
        self.label_column =label_column 
        self.reference_label = reference_label
        self.aggregrate_prune =  aggregrate_prune
        self.data_column = data_column

        # Load or copy
        if isinstance(data_source, (str, Path)):
            self._df = pd.read_csv(data_source)
        else:
            self._df = data_source.copy()

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if not pipeline:
            self._pipeline = [EmbeddingPruner(self._df)]
        self._pipeline = pipeline

    def __call__(self) -> pd.DataFrame:
        """
        Run each pruner in sequence, passing the DataFrame result from one into the next.
        Before running each pruner, copy the latest “*_accept” column into `prev_accept`.
        """
        df = self._df

        for pruner in self._pipeline:
            result_df = pruner(df, 
                               self._output_dir, 
                               self.label_column,  
                               self.reference_label, 
                               self.data_column)
            
            if self.aggregrate_prune:
                df = result_df[result_df[pruner.NAME]]

        df.to_csv(self._output_dir / "squirrel_pruned.csv", index=False)
        return df

