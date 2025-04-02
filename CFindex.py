import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y

def conn_index_matrix(X, labels, n_neighbors=1):
    """
    Compute the label-pairwise CONN index matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.

    labels : array-like of shape (n_samples,)
        Cluster labels for each sample.

    n_neighbors : int, default=1
        Number of nearest neighbors to consider.

    Returns
    -------
    conn_matrix : ndarray of shape (n_labels, n_labels)
        Matrix where entry (i, j) is the average number of neighbors of points
        in cluster i that are in cluster j.
    """
    X, labels = check_X_y(X, labels)
    n_samples = X.shape[0]

    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be less than the number of samples.")

    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    n_labels = len(unique_labels)

    conn_matrix = np.zeros((n_labels, n_labels), dtype=float)
    cluster_counts = np.zeros(n_labels, dtype=int)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # Exclude self

    for i in range(n_samples):
        label_i = labels[i]
        idx_i = label_to_index[label_i]
        cluster_counts[idx_i] += 1

        for j in neighbor_indices[i]:
            label_j = labels[j]
            idx_j = label_to_index[label_j]
            conn_matrix[idx_i, idx_j] += 1

    # Normalize by (number of samples in cluster i * k)
    for idx in range(n_labels):
        if cluster_counts[idx] > 0:
            conn_matrix[idx] /= (cluster_counts[idx] * n_neighbors)

    return conn_matrix


import numpy as np


class CFIndex:
    def __init__(self):
        self.X_all = []
        self.y_all = []
        self.task_preds = []  # List of predicted labels after each task
        self.task_true_labels = []  # List of true labels for each task
        self.task_ids = []  # List of task indices aligned with X and y

    def add_task(self, X, y_true, y_pred, task_id):
        """
        Store a new task's data and predictions.

        Parameters:
        - X: Samples for this task
        - y_true: Ground truth labels
        - y_pred: Predictions made after this task was learned
        - task_id: Integer task identifier
        """
        self.X_all.append(X)
        self.y_all.append(y_true)
        self.task_preds.append(y_pred)
        self.task_true_labels.append(y_true)
        self.task_ids.append(task_id)

    def compute(self):
        """
        Compute the OM, forgetting scores, and corrected CF scores.

        Returns:
            - corrected_CF_scores: List of corrected CF scores per task
            - avg_CF_index: Mean corrected CF score
        """
        X_full = np.vstack(self.X_all)
        y_full = np.concatenate(self.y_all)

        # Compute the directed Overlap Metric matrix
        OM = conn_index_matrix(X_full, y_full)  # Should return shape (T, T)

        T = len(self.task_true_labels)
        acc_matrix = np.zeros((T, T))

        for i in range(T):
            y_true_i = self.task_true_labels[i]
            for j in range(i, T):
                y_pred_i_after_j = self.task_preds[j][self.task_ids[i] == i]
                acc_matrix[i, j] = np.mean(y_pred_i_after_j == y_true_i)

        # Forgetting per task
        forgetting_scores = np.max(acc_matrix[:, :-1], axis=1) - acc_matrix[:, -1]

        # Compute corrected CF scores with OM-based weighting
        corrected_CF_scores = []
        for i in range(T):
            future_overlap = [OM[i, j] for j in range(i + 1, T)]
            if not future_overlap:
                corrected_CF_scores.append(0.0)
                continue
            min_om = min(future_overlap)

            if min_om < 0.90:
                w = 0.0
            elif min_om < 0.99:
                w = (min_om - 0.90) / 0.099
            else:
                w = 1.0

            corrected_CF_scores.append(w * forgetting_scores[i])

        avg_CF_index = np.mean(corrected_CF_scores)
        return corrected_CF_scores, avg_CF_index
