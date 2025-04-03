import numpy as np
from artlib import FuzzyART, SimpleARTMAP
from typing import Optional
from collections import defaultdict

class GrowingSquareArray:
    def __init__(self):
        self.array = np.zeros((0, 0), dtype=int)

    def _ensure_size(self, i, j):
        size = max(i + 1, j + 1)
        if size > self.array.shape[0]:  # Always square
            new_array = np.zeros((size, size), dtype=self.array.dtype)
            if self.array.size > 0:
                new_array[:self.array.shape[0], :self.array.shape[1]] = self.array
            self.array = new_array

    def __getitem__(self, idx):
        i, j = idx
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
            T[c1_] = -np.inf
            c2_ = int(np.argmax(T))
        else:
            c2_ = 1
        return c1_, c2_

class CONNSimplifiedARTMAP(SimpleARTMAP):
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

class iCONN:
    def __init__(self, rho: float = 0.9):
        module_a = CONNFuzzyART(rho=rho, alpha=0.0, beta=1.0)
        self.FuzzyARTMAP = SimpleARTMAP(module_a)
        self.CADJ = GrowingSquareArray()
        self.CONN = GrowingSquareArray()
        self.INTRA = GrowingArray1D()
        self.inter_conn = 0.0
        self.INTER = GrowingSquareArray()
        self.cluster_cardinality = GrowingArray1D()
        self.rev_map = defaultdict(set)
        self.index = np.nan

    def add_sample(self, x, y):
        self.FuzzyARTMAP = self.FuzzyARTMAP.partial_fit([x], [y])
        bmu1 = self.FuzzyARTMAP.module_a.labels_[-1]
        c_a1_, c_a2_ = self.FuzzyARTMAP.module_a.step_pred_first_and_second(x)
        bmu2 = (c_a2_ if bmu1 == c_a1_ else c_a1_)
        self.rev_map[y].add(bmu1)

        self.cluster_cardinality[y] += 1
        self.CADJ[bmu1, bmu2] += 1
        self.CONN[bmu1, bmu2] = self.CADJ[bmu1, bmu2] + self.CADJ[bmu2, bmu1]

        assert y == self.FuzzyARTMAP.map[bmu1]
        if bmu2 not in self.FuzzyARTMAP.map:
            y2 = int(y)
        else:
            y2 = self.FuzzyARTMAP.map[bmu2]

        ### THIS PROBABLY NEEDS TO CHANGE
        if y == y2:
            # same class
            intra_old = float(self.INTRA[y])
            S = np.array(sorted(self.rev_map[y]))
            self.INTRA[y] = self.CADJ[np.ix_(S, S)].sum() / self.cluster_cardinality[y]
            self.intra_conn = sum(self.INTRA)/len(self.rev_map)
        else:
            # different classes
            S1 = np.array(sorted(self.rev_map[y]))
            S2 = np.array(sorted(self.rev_map[y2]))
            inter_numer = self.CONN[np.ix_(S1, S2)].sum()

            # Mask to select CADJ[i, j] > 0 for i in S1 and j in S2
            CADJ_sub = self.CADJ[np.ix_(S1, S2)]
            CONN_sub = self.CONN[np.ix_(S1, S2)]
            # Find which rows in CADJ_sub have any positive value
            valid_rows = np.any(CADJ_sub > 0, axis=1)
            # Filter CONN_sub to only those rows
            inter_denom = np.sum(CONN_sub[valid_rows])
            # update INTER
            self.INTER[y, y2] = inter_numer / inter_denom

            A = self.INTER.asarray()
            self.inter_conn = np.max(
                np.where(
                    ~np.eye(
                        A.shape[0],
                        dtype=bool
                    ),
                    A,
                    -np.inf
                ),
                axis=1
            ).sum() / len(self.rev_map)
        self.index = self.intra_conn * (1-self.inter_conn)


