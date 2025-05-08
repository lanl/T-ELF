import json
import logging
import math
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

from ....helpers.embeddings import compute_embeddings

log = logging.getLogger(__name__)

class EmbeddingPruner:
    """
    Prune documents by distance from a reference-class centroid in embedding space
    """

    def __init__(
        self,
        *,
        embedding_model: str = "SCINCL",
        distance_std_factor: float = 3.0,
        overwrite_embeddings: bool = False,
        use_gpu: Optional[bool] = None,
        verbose: bool = True,
    ):
        """
        Initialize the EmbeddingPruner.

        Parameters
        ----------
        embedding_model : str
            Name of the embedding model to use.
        distance_std_factor : float
            Multiplier on standard deviation to set distance threshold.
        overwrite_embeddings : bool
            If True, always recompute embeddings even if cache exists.
        use_gpu : bool or None
            Whether to use GPU for embedding. If None, auto-detect.
        verbose : bool
            Whether to display progress bars during embedding.
        """
        self.embedding_model = embedding_model
        self.distance_std_factor = distance_std_factor
        self.overwrite_embeddings = overwrite_embeddings
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.verbose = verbose
        self.NAME = 'embed_prune'

    def load_or_compute_embeddings(self, df, output_dir, data_column) -> np.ndarray:
        """
        Load or compute embeddings for the specified column in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be processed.
        output_dir : str or Path
            Directory to save the output files.
        data_column : str
            Column name containing the data to be voted on.
        """
        cache_path = Path(output_dir) / "embeddings.npy"
        if cache_path.exists() and not self.overwrite_embeddings:
            emb = np.load(cache_path)
            if emb.shape[0] == len(df):
                return emb
            log.warning("Cache size %d != rows %d, recomputing", emb.shape[0], len(df))

        # Batch compute embeddings
        def batch_embed(df_slice: pd.DataFrame) -> np.ndarray:
            return compute_embeddings(
                df_slice,
                model_name=self.embedding_model,
                cols=[data_column],
                sep_token="[SEP]",
                as_np=True,
                use_gpu=self.use_gpu
            ).astype(np.float32)

        try:
            emb = batch_embed(df)
        except Exception as e:
            log.warning("Batch embed failed (%s), row-wise fallback", e)
            emb = np.vstack([
                batch_embed(df.iloc[i:i+1])[0][None, :] 
                for i in tqdm(range(len(df)), disable=not self.verbose)
            ])

        if emb.shape[0] != len(df):
            raise ValueError(f"Embeddings {emb.shape[0]} != rows {len(df)}")

        np.save(cache_path, emb)
        return emb

    def select_inliers(self, df, emb: np.ndarray, label_column: str, reference_label: Union[int, str]) -> np.ndarray:
        """
        Compute which rows are within threshold distance to reference centroid

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the dataset.
        emb : np.ndarray
            Embedding matrix of shape (n_samples, embedding_dim).
        label_column : str
            Column name indicating class labels.
        reference_label : int | str
            Label used as the reference class for centroid.

        Returns
        -------
        inliers_mask : np.ndarray of bool
            Mask indicating rows within distance threshold.
        """

        # Compute centroid only on previously accepted
        cluster_mask = df[label_column] == reference_label
        centroid = emb[cluster_mask].mean(axis=0)

        # Distances for all
        dists = np.linalg.norm(emb - centroid, axis=1)
        mu, sigma = dists[cluster_mask].mean(), dists[cluster_mask].std()
        thresh = mu + self.distance_std_factor * sigma
        # Inliers
        inliers = (dists <= thresh) 
        log.info("Embed prune: thr=%.4f, kept %d ", thresh, inliers.sum() )
        return inliers

    def __call__(self, df, output_dir, label_column: str, reference_label: Union[int, str], data_column:str) -> pd.DataFrame:
        """
        Execute pruning: annotate 'embed_accept' for rows that were inliers.
        Saves the annotated DataFrame to CSV and returns it.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be processed.
        output_dir : str or Path
            Directory to save the output files.
        label_column : str
            Column name indicating class labels.
        reference_label : Union[int, str]
            Label used as the reference class for centroid.
        data_column : str
            Column name containing the data to be voted on.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with added 'embed_accept' column.
        """
        
        output_dir = Path(output_dir) / "embed_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        emb = self.load_or_compute_embeddings(df, output_dir, data_column)
        inliers = self.select_inliers(df, emb, label_column, reference_label)
        df[self.NAME] = inliers

        out_csv = Path(output_dir) / "embed_pruned.csv"
        df.to_csv(out_csv, index=False)
        log.info("Saved embed-pruned DF → %s", out_csv)

        return df
