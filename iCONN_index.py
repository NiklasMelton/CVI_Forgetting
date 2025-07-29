import numpy as np
from artlib import FuzzyART, SimpleARTMAP
from artlib.common.utils import complement_code
from typing import Optional, Literal
from collections import defaultdict
import numbers
from iKMeans import iKMeans

class GrowingSquareArray:
    def __init__(self, dtype=int):
        self.array = np.zeros((0, 0), dtype=dtype)

    def _ensure_size(self, i, j):
        size = max(i + 1, j + 1)
        if size > self.array.shape[0]:  # Always square
            new_array = np.zeros((size, size), dtype=self.array.dtype)
            if self.array.size > 0:
                new_array[:self.array.shape[0], :self.array.shape[1]] = self.array
            self.array = new_array

    def __getitem__(self, idx):
        i, j = idx
        # if either index is an array/slice, forward directly to the numpy array:
        if (not isinstance(i, numbers.Integral)
            or not isinstance(j, numbers.Integral)):
            return self.array[idx]

        # otherwise do your auto-resize + element access
        self._ensure_size(i, j)
        return self.array[i, j]

    def __setitem__(self, idx, value):
        i, j = idx
        self._ensure_size(i, j)
        self.array[i, j] = value

    def __iadd__(self, idx_value):
        idx, value = idx_value
        i, j = idx
        self._ensure_size(i, j)
        self.array[i, j] += value
        return self

    def __repr__(self):
        return repr(self.array)

    def asarray(self):
        return self.array.copy()


class GrowingArray1D:
    def __init__(self, dtype=int):
        self.array = np.zeros(0, dtype=dtype)

    def _ensure_size(self, i):
        if i >= self.array.size:
            new_size = i + 1
            new_array = np.zeros(new_size, dtype=self.array.dtype)
            new_array[:self.array.size] = self.array
            self.array = new_array

    def __getitem__(self, i):
        self._ensure_size(i)
        return self.array[i]

    def __setitem__(self, i, value):
        self._ensure_size(i)
        self.array[i] = value

    def __iadd__(self, idx_value):
        i, value = idx_value
        self._ensure_size(i)
        self.array[i] += value
        return self

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return repr(self.array)

    def asarray(self):
        return self.array.copy()

    def __iter__(self):
        # iterate over the *current* contents only
        for v in self.array:
            yield v

class CONNFuzzyART(FuzzyART):
    def step_pred_first_and_second(self, x) -> int:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        assert len(self.W) >= 0, "ART module is not fit."

        T, _ = zip(*[self.category_choice(x, w, params=self.params) for w in self.W])
        c1_ = int(np.argmax(T))
        if len(T) > 1:
            T = list(T)
            T[c1_] = -np.inf
            c2_ = int(np.argmax(T))
        else:
            c2_ = 1
        return c1_, c2_

class CONNiKMeans(iKMeans):
    def step_pred_first_and_second(self, x) -> int:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        assert len(self.cluster_centers_) >= 0, "iKmeans module is not fit."

        T = self.get_activation(x.reshape((1,-1))).reshape((-1))
        c1_ = int(np.argmax(T))
        if len(T) > 1:
            T = list(T)
            T[c1_] = -np.inf
            c2_ = int(np.argmax(T))
        else:
            c2_ = 1
        return c1_, c2_


class CONNSimpleARTMAP(SimpleARTMAP):
    def match_reset_func(
        self,
        i: np.ndarray,
        w: np.ndarray,
        cluster_a,
        params: dict,
        extra: dict,
        cache: Optional[dict] = None,
    ) -> bool:
        """Permits external factors to influence cluster creation.

        Parameters
        ----------
        i : np.ndarray
            Data sample.
        w : np.ndarray
            Cluster weight / info.
        cluster_a : int
            A-side cluster label.
        params : dict
            Parameters for the algorithm.
        extra : dict
            Additional parameters, including "cluster_b".
        cache : dict, optional
            Values cached from previous calculations.

        Returns
        -------
        bool
            True if the match is permitted, False otherwise.

        """

        cluster_b = extra["cluster_b"]
        b_samples = sum(
            self.module_a.weight_sample_counter_[a]
            for a, b in self.map.items()
            if b == cluster_b
        )
        if b_samples == 1:
            return False
        if cluster_a in self.map and self.map[cluster_a] != cluster_b:
            return False
        return True

    def step_pred_first_and_second(self, x) -> int:
        """Predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            Data sample.

        Returns
        -------
        int
            Cluster label of the input sample.

        """
        return self.module_a.step_pred_first_and_second(x)

class iCONN:
    def __init__(self, method: Literal["Fuzzy", "Kmeans"] = "Fuzzy", rho: float = 0.9, k: int = 10, epsilon: float = 1e-1):
        assert method in ["Fuzzy", "Kmeans"]
        self.method = method
        if self.method == "Fuzzy":
            module_a = CONNFuzzyART(rho=rho, alpha=1e-10, beta=1.0)
            self.model = CONNSimpleARTMAP(module_a)
        else:
            self.model = CONNiKMeans(K=k, epsilon=epsilon)

        self.CADJ = GrowingSquareArray()
        self.CONN = GrowingSquareArray()
        self.INTRA = GrowingArray1D(dtype=float)
        self.inter_conn = 0.0
        self.INTER = GrowingSquareArray(dtype=float)
        self.cluster_cardinality = GrowingArray1D()
        self.rev_map = defaultdict(set)
        self.index = np.nan

    def _calc_inter(self, i, j):
        S1 = np.array(sorted(self.rev_map[i]), dtype=int)
        S2 = np.array(sorted(self.rev_map[j]), dtype=int)
        inter_numer = self.CONN[np.ix_(S1, S2)].sum()

        # Mask to select CADJ[i, j] > 0 for i in S1 and j in S2
        CADJ_sub = self.CADJ[np.ix_(S1, S2)]
        CONN_sub = self.CONN[np.ix_(S1, S2)]
        # Find which rows in CADJ_sub have any positive value
        valid_rows = np.any(CADJ_sub > 0, axis=1)
        # Filter CONN_sub to only those rows
        inter_denom = np.sum(CONN_sub[valid_rows])
        # calculate INTER
        if inter_denom == 0:
            return 0.0
        return inter_numer / inter_denom

    def _update_metric(self, y, y2):
        S = np.array(sorted(self.rev_map[y]))
        self.INTRA[y] = self.CADJ[np.ix_(S, S)].sum() / self.cluster_cardinality[y]
        self.intra_conn = sum(self.INTRA) / len(self.rev_map)
        if y != y2:
            # different classes
            self.INTER[y, y2] = self._calc_inter(y, y2)
            self.INTER[y2, y] = self._calc_inter(y2, y)
        else:
            # same class
            for m in self.rev_map.keys():
                if m != y:
                    self.INTER[y, m] = self._calc_inter(y, m)
        A = self.INTER.asarray()
        n = A.shape[0]  # should equal len(self.rev_map)
        if n < 2:
            # with fewer than two classes, define inter-conn as 0
            self.inter_conn = 0.0
        else:
            # compute off-diagonal maxes
            mask = ~np.eye(n, dtype=bool)
            off_diag = np.where(mask, A, -np.inf)
            # max over each row, sum, normalize
            self.inter_conn = np.max(off_diag, axis=1).sum() / n

        self.index = self.intra_conn * (1 - self.inter_conn)


    def add_sample(self, x, y):
        x_prep = complement_code([x])
        self.model = self.model.partial_fit(x_prep, [y], match_tracking="MT~")
        if self.method == "Fuzzy":
            bmu1 = self.model.module_a.labels_[-1]
        else:
            bmu1 = self.sample_cluster_ids_[-1]
        c_a1_, c_a2_ = self.model.step_pred_first_and_second(x)
        bmu2 = (c_a2_ if bmu1 == c_a1_ else c_a1_)
        self.rev_map[y].add(bmu1)

        self.cluster_cardinality[y] += 1
        self.CADJ[bmu1, bmu2] += 1
        self.CONN[bmu1, bmu2] = self.CADJ[bmu1, bmu2] + self.CADJ[bmu2, bmu1]

        if self.method == "Fuzzy":
            if bmu2 not in self.model.map:
                y2 = int(y)
            else:
                y2 = self.model.map[bmu2]
        else:
            if bmu2 not in self.model.cluster_id_to_class_:
                y2 = int(y)
            else:
                y2 = self.model.cluster_id_to_class_[bmu2]

        self._update_metric(y, y2)
        return self.index

    def add_batch(self, X, Y):
        if self.method == "Fuzzy":
            X_prep = complement_code(X)
        else:
            X_prep = X
        self.model = self.model.partial_fit(X_prep, Y, match_tracking="MT~")
        if self.method == "Fuzzy":
            BMU1 = self.model.module_a.labels_[-len(Y):]
        else:
            BMU1 = self.model.cluster_labels_[-len(Y):]

        for x, y, bmu1 in zip(X_prep, Y, BMU1):
            y = int(y)
            c_a1_, c_a2_ = self.model.step_pred_first_and_second(x)
            bmu2 = (c_a2_ if bmu1 == c_a1_ else c_a1_)
            self.rev_map[y].add(bmu1)

            self.cluster_cardinality[y] += 1
            self.CADJ[bmu1, bmu2] += 1
            self.CONN[bmu1, bmu2] = self.CADJ[bmu1, bmu2] + self.CADJ[bmu2, bmu1]

            if self.method == "Fuzzy":
                assert y == self.model.map[bmu1]
                if bmu2 not in self.model.map:
                    y2 = int(y)
                else:
                    y2 = int(self.model.map[bmu2])
            else:
                if bmu2 not in self.model.cluster_id_to_class_:
                    y2 = int(y)
                else:
                    y2 = self.model.cluster_id_to_class_[bmu2]

            self._update_metric(y, y2)
        return self.index

