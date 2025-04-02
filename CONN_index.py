import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y

def conn_score(X, labels, n_neighbors=1):
    """
    Compute the CONN (Connectivity) index for evaluating clustering quality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    labels : array-like of shape (n_samples,)
        Cluster labels for each sample.

    n_neighbors : int, default=1
        Number of nearest neighbors to consider (commonly 1).

    Returns
    -------
    conn_score : float
        The CONN index score, ranging from 0 to 1. Higher is better.
    """
    X, labels = check_X_y(X, labels)
    n_samples = X.shape[0]

    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than the number of samples.")

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)

    neighbor_indices = indices[:, 1:]  # Exclude self-neighbor

    match_count = 0
    for i in range(n_samples):
        for j in neighbor_indices[i]:
            if labels[i] == labels[j]:
                match_count += 1

    conn_score = match_count / (n_samples * n_neighbors)
    return conn_score
