from OverlapIndex import OverlapIndex
import numpy as np
from typing import Literal, Optional, Dict, Tuple
from collections import defaultdict


def compute_recall(y_true, y_pred):
    counts = defaultdict(int)
    correct = defaultdict(int)
    for yt, yp in zip(y_true, y_pred):
        counts[yt] += 1
        if yt == yp:
            correct[yt] += 1
    return {c: correct[c] / counts[c] for c in counts}

def compute_mean_scores(scores, y_true):
    mean_scores = defaultdict(float)
    for c in range(scores.shape[1]):
        class_scores = scores[:, c]
        true_scores = class_scores[y_true==c]
        mean_scores[c] = float(np.mean(true_scores))
    return mean_scores


class OFI:
    def __init__(
            self,
            rho: float = 0.9,
            r_hat: float = np.inf,
            ART: Literal["Fuzzy", "Hypersphere"] = "Hypersphere",
            match_tracking = "MT+",
    ):
        self.OI = OverlapIndex(rho=rho, r_hat=r_hat, ART=ART, match_tracking=match_tracking)
        self.global_indices = {"overshadowing": 0.0, "forgetting": 0.0}
        self.cluster_indices = defaultdict(lambda: {"overshadowing": 0.0, "forgetting": 0.0})
        self.max_scores: Dict[int, float] = defaultdict(lambda: -np.inf)
        self.min_scores: Dict[int, float] = defaultdict(lambda: np.inf)
        self.max_recalls: Dict[int, float] = defaultdict(float)


    def add_batch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_pred_eval: np.ndarray,
        y_true_eval: np.ndarray,
        y_scores_eval: np.ndarray

    ) -> Tuple[float, float]:
        """
        Update the OCF with a training batch and evaluation result.

        (overshadowing, forgetting, residual) as floats in [0, 1]
        """
        # 0. check if we lost any classes in the evaluation set
        historical = set(self.max_recalls.keys())
        current = set(int(x) for x in y_true_eval)  # ensure plain ints
        missing = historical - current
        assert not missing, f"Missing classes in eval: {missing}"

        # 1. Compute oi index from training batch
        _ = self.OI.add_batch(X_train, y_train)


        # 2. Compute recall and score changes per class
        curr_recalls = compute_recall(y_true_eval, y_pred_eval)
        mean_scores = compute_mean_scores(y_scores_eval, y_true_eval) # eq 7

        recall_drops = defaultdict(float)
        score_drops = defaultdict(float)

        for cls, curr_r in curr_recalls.items():
            prev_best = self.max_recalls[cls]
            self.max_recalls[cls] = max(prev_best, curr_r)
            recall_drops[cls] = self.max_recalls[cls] - curr_r # eq 8

            # allow for rebalancing if recall isn't degraded # eq 9 & 10
            if curr_r >= prev_best:
                self.max_scores[cls] = mean_scores[cls]
                if np.isinf(self.min_scores[cls]):
                    self.min_scores[cls] = mean_scores[cls]
            else:
                self.max_scores[cls] = max(self.max_scores[cls], mean_scores[cls])
                self.min_scores[cls] = min(self.min_scores[cls], mean_scores[cls])
            score_span = self.max_scores[cls] - self.min_scores[cls]
            if score_span > 0:
                score_drops[cls] = (
                    self.max_scores[cls] - mean_scores[cls] # eq 11
                )/score_span
            else:
                score_drops[cls] = 0.0


        # 3. compute indices
        Fs: Dict[int, float] = {
            c: recall_drops[c]*score_drops[c]
            for c in recall_drops
        } # eq 12


        Os: Dict[int, float] = {
            c: recall_drops[c]*(
                (1.-self.OI.singleton_index[c])
            )*(1-score_drops[c])
            for c in recall_drops.keys()
        } # eq 13

        for c in recall_drops:
            FO_sum = Fs[c]+Os[c] # eq 14
            if FO_sum > 0:
                ratio = recall_drops[c]/FO_sum
                Fs[c] = ratio*Fs[c] # eq 15
                Os[c] = ratio*Os[c] # eq 16

        F = float(np.mean(list(Fs.values()))) # eq 17
        O = float(np.mean(list(Os.values()))) # eq 18

        for c in curr_recalls.keys():
            self.cluster_indices[c]["overshadowing"] = Os[c]
            self.cluster_indices[c]["forgetting"] = Fs[c]

        self.global_indices["overshadowing"] = O
        self.global_indices["forgetting"] = F

        return O, F
    
