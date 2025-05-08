import umap
import numpy as np

def get_umap_reducer(
    n_neighbors: int,
    min_dist: float,
    metric: str = "cosine",
    random_state: int = 42
) -> umap.UMAP:
    """
    Initialize and return a UMAP reducer with specified parameters.
    """
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )


def compute_umap(
    reducer: umap.UMAP,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using the provided UMAP reducer.

    Returns
    -------
    np.ndarray
        2D coordinates for each sample.
    """
    return reducer.fit_transform(embeddings)
