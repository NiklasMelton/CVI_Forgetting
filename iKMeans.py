import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
from typing import Optional

class iKMeans:
    def __init__(self, K: int, epsilon: float = 1e-5):
        """
        Parameters
        ----------
        K : int
            Total number of clusters to assign for each batch.
        epsilon : float
            Distance threshold below which clusters are merged.
        """
        self.K = K
        self.epsilon = epsilon
        self.cluster_centers_ = []
        self.cluster_labels_ = []
        self.sample_cluster_ids_ = []  # Global cluster index assigned to each sample
        self.sample_true_labels_ = []  # Ground-truth label for each sample
        self.class_to_cluster_ids_ = defaultdict(list)
        self.cluster_id_to_class_ = dict()
        self.total_clusters = 0

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit KMeans separately for each class and integrate results into global model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for the current batch.
        y : ndarray of shape (n_samples,)
            True labels for each sample.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples."

        unique_classes, counts = np.unique(y, return_counts=True)

        for cls in unique_classes:
            class_mask = (y == cls)
            X_class = X[class_mask]
            y_class = y[class_mask]
            k_cls = min(self.K, len(X_class))

            kmeans = KMeans(n_clusters=k_cls, n_init="auto", algorithm="lloyd")
            kmeans.fit(X_class)
            centers = kmeans.cluster_centers_

            self.cluster_centers_.extend(centers)
            self.cluster_labels_.extend([cls] * k_cls)

            # Store mapping of each sample to cluster ID
            cluster_ids = kmeans.predict(X_class) + self.total_clusters
            self.sample_cluster_ids_.extend(cluster_ids)
            self.sample_true_labels_.extend(y_class.tolist())

            # Update cluster-to-class mapping
            for cid in range(k_cls):
                global_cid = self.total_clusters + cid
                self.cluster_id_to_class_[global_cid] = cls

            # Update class-to-cluster mapping
            for cid in range(k_cls):
                self.class_to_cluster_ids_[cls].append(self.total_clusters + cid)

            self.total_clusters += k_cls
        self._merge_nearby_clusters()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class label for each sample based on nearest cluster.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if not self.cluster_centers_:
            raise ValueError("No clusters available. Fit at least one batch first.")

        centers = np.vstack(self.cluster_centers_)
        distances = cdist(X, centers)
        closest_cluster = np.argmin(distances, axis=1)
        return np.array([self.cluster_labels_[cid] for cid in closest_cluster])

    def get_activation(self, X: np.ndarray) -> np.ndarray:
        """
        Return negative distances to each cluster center for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        activation : ndarray of shape (n_samples, n_clusters)
            Negative Euclidean distances to cluster centers.
        """
        if not self.cluster_centers_:
            raise ValueError("No clusters available.")
        centers = np.vstack(self.cluster_centers_)
        distances = cdist(X, centers)
        return -distances

    def get_sample_assignments(self):
        """
        Return the stored cluster and true labels for all seen samples.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - cluster_ids: Global cluster indices for all samples
            - true_labels: Corresponding ground truth labels
        """
        return np.array(self.sample_cluster_ids_), np.array(self.sample_true_labels_)

    def _merge_nearby_clusters(self):
        """
        Merge clusters whose centers are within epsilon distance.
        Updates cluster_centers_, cluster_labels_, mappings, and assignments.
        """
        if len(self.cluster_centers_) < 2:
            return

        centers = np.vstack(self.cluster_centers_)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance

        merged = np.full(len(centers), -1, dtype=int)
        new_centers = []
        new_labels = []
        cid_map = {}  # old index → new index

        for i in range(len(centers)):
            if merged[i] != -1:
                continue
            group = [i]
            for j in range(i + 1, len(centers)):
                if dist_matrix[i, j] < self.epsilon and self.cluster_labels_[i] == \
                        self.cluster_labels_[j]:
                    group.append(j)
                    merged[j] = i
            group_centers = centers[group]
            new_center = np.mean(group_centers, axis=0)
            new_cid = len(new_centers)
            for old_cid in group:
                cid_map[old_cid] = new_cid
            new_centers.append(new_center)
            new_labels.append(self.cluster_labels_[i])

        # Update attributes
        self.cluster_centers_ = new_centers
        self.cluster_labels_ = new_labels
        self.total_clusters = len(new_centers)

        # Remap sample cluster IDs
        self.sample_cluster_ids_ = [cid_map[cid] for cid in self.sample_cluster_ids_]

        # Rebuild cluster_id_to_class_ and class_to_cluster_ids_
        self.cluster_id_to_class_.clear()
        self.class_to_cluster_ids_.clear()
        for new_cid, cls in enumerate(new_labels):
            self.cluster_id_to_class_[new_cid] = cls
            self.class_to_cluster_ids_[cls].append(new_cid)

