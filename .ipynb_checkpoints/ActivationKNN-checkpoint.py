from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import numpy as np

class KNN:
    """
    KNN classifier with an activation function per class:
    activation(sample, class) = - min_distance(sample, any training point of that class)
    """

    def __init__(self, n_neighbors=5, **knn_kwargs):
        # standard KNN for prediction
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, **knn_kwargs)
        # per-class nearest-neighbor models for activation
        self.class_nn = {}
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the classifier and build per-class 1-NN models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.knn.fit(X, y)
        self.classes_ = self.knn.classes_

        # Build a NearestNeighbors(1) model for each class
        for cls in self.classes_:
            X_cls = X[y == cls]
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(X_cls)
            self.class_nn[cls] = nn

        return self

    def predict(self, X):
        """
        Standard KNN predict.
        """
        return self.knn.predict(X)

    def activation(self, X):
        """
        Compute activations for each sample and each class:
          activations[i, j] = - min_distance(X[i], any training point of class j)

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        activations : ndarray of shape (n_queries, n_classes)
            Column j corresponds to self.classes_[j].
        """
        X = np.atleast_2d(X)
        n_queries = X.shape[0]
        n_classes = len(self.classes_)
        activations = np.zeros((n_queries, n_classes))

        for idx, cls in enumerate(self.classes_):
            nn = self.class_nn[cls]
            distances, _ = nn.kneighbors(X, n_neighbors=1)
            activations[:, idx] = -distances.ravel()

        return activations
