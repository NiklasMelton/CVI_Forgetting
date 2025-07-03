import numpy as np
from sklearn.utils.validation import check_X_y


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
