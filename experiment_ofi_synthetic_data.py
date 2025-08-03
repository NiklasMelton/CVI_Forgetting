import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Dict

from OFI import OFI
from synthetic_datasets import generate_synthetic_blobs
from ActivationKNN import KNN

from common import make_dirs


def experiment_ofi_synthetic_data_cnn():

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    def build_mlp(input_dim: int = 2, n_classes: int = 3
                  ) -> Tuple[keras.Model, keras.Model]:
        init = keras.initializers.GlorotUniform(seed=SEED)
        inp = keras.Input(shape=(input_dim,), name="input")
        x = keras.layers.Dense(16, activation="relu", kernel_initializer=init,
                               name="dense1")(inp)
        x = keras.layers.Dense(16, activation="relu", kernel_initializer=init,
                               name="dense2")(x)
        logits = keras.layers.Dense(n_classes, activation=None, kernel_initializer=init,
                                    name="logits")(x)
        probs = keras.layers.Activation("softmax", name="softmax")(logits)

        train_model = keras.Model(inp, probs, name="classifier")
        logit_model = keras.Model(inp, logits, name="logit_extractor")

        train_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-2),
            loss="sparse_categorical_crossentropy",
            metrics=[]
        )
        return train_model, logit_model

    def run_condition(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            batch_size: int = 50,
            rho: float = 0.9,
            r_hat: float = 0.1,
            ART: str = "Fuzzy"
    ) -> Tuple[
        List[List[float]],
        List[float],
        List[Tuple[float, float]]
    ]:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .batch(batch_size, drop_remainder=True)

        x_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.int32)

        model, logit_extractor = build_mlp(X_train.shape[1], int(np.max(y_train)) + 1)
        cf_detector = OFI(rho=rho, r_hat=r_hat, ART=ART)
        cf_detector.OI.ARTMAP.module_a.set_data_bounds(np.zeros((2,)), np.ones((2,)))

        tpr_trace: List[List[float]] = []
        oi_trace: List[float] = []
        ofi_trace: List[Tuple[float, float]] = []

        for x_b, y_b in train_ds:
            model.train_on_batch(x_b, y_b)

            logits = logit_extractor.predict(x_test_tf, batch_size=y_test_tf.shape[0],
                                             verbose=0)
            probs = model.predict(x_test_tf, batch_size=y_test_tf.shape[0], verbose=0)
            y_pred = np.argmax(probs, axis=1)

            tprs = []
            for cls in range(probs.shape[1]):
                mask = (y_test_tf.numpy() == cls)
                tp = np.sum(y_pred[mask] == cls)
                fn = np.sum(y_pred[mask] != cls)
                tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            tpr_trace.append(tprs)

            x_b_prep = cf_detector.OI.ARTMAP.module_a.prepare_data(x_b.numpy())
            O, F = cf_detector.add_batch(
                X_train=x_b_prep,
                y_train=y_b.numpy(),
                y_pred_eval=y_pred,
                y_true_eval=y_test,
                y_scores_eval=logits
            )
            ofi_trace.append((O, F))
            oi_trace.append(cf_detector.OI.index)

        return tpr_trace, oi_trace, ofi_trace


    traces_tpr: Dict[str, List[List[float]]] = {}
    traces_oi: Dict[str, List[float]] = {}
    traces_state: Dict[str, List[Tuple[float, float]]] = {}

    for ov, od in ((False, False), (False, True), (True, False), (True, True)):
        X_tr, y_tr, X_te, y_te = generate_synthetic_blobs(
            n_samples=3600,
            test_size=600,
            overlap=ov,
            ordered=od,
            random_state=SEED
        )
        cond = ("Overlapped, " if ov else "Separated, ") + (
            "Ordered" if od else "Shuffled")
        print(f"→ running condition {cond}")
        tpr, oi_idx, ofi_states = run_condition(
            X_tr, y_tr, X_te, y_te,
            batch_size=20,
            rho=0.95,
            r_hat=0.1,
            ART="Fuzzy"
        )
        traces_tpr[cond] = tpr
        traces_oi[cond] = oi_idx
        traces_state[cond] = ofi_states

    # Save as structured arrays for traceability
    path = "results_data/OFI/synthetic/ofi_cnn_traces_synthetic_data.npz"
    make_dirs(path)
    np.savez(
        path,
        ov_od_tpr=traces_tpr["Overlapped, Ordered"],
        ov_sh_tpr=traces_tpr["Overlapped, Shuffled"],
        sp_od_tpr=traces_tpr["Separated, Ordered"],
        sp_sh_tpr=traces_tpr["Separated, Shuffled"],
        ov_od_int=traces_oi["Overlapped, Ordered"],
        ov_sh_int=traces_oi["Overlapped, Shuffled"],
        sp_od_int=traces_oi["Separated, Ordered"],
        sp_sh_int=traces_oi["Separated, Shuffled"],
        ov_od_states=np.array(traces_state["Overlapped, Ordered"]),
        ov_sh_states=np.array(traces_state["Overlapped, Shuffled"]),
        sp_od_states=np.array(traces_state["Separated, Ordered"]),
        sp_sh_states=np.array(traces_state["Separated, Shuffled"]),
    )

    print(
        "Saved per-batch TPR, OverlapIndex, and OFI states to cf2_batch_traces.npz")


def experiment_ofi_synthetic_data_knn():

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    def run_condition(
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            batch_size: int = 50,
            rho: float = 0.9,
            r_hat: float = 0.1,
            ART: str = "Fuzzy",
            knn_kwargs: dict = None
    ) -> Tuple[
        List[List[float]],  # tpr_trace
        List[float],  # oi_trace
        List[Tuple[float, float]],  # global ofi_trace
        List[List[float]],  # per-cluster forgetting
        List[List[float]]  # per-cluster overshadowing
    ]:
        if knn_kwargs is None:
            knn_kwargs = {"n_neighbors": 1}

        n_true = int(np.max(y_train)) + 1
        X_mem: List[np.ndarray] = []
        y_mem: List[np.ndarray] = []

        knn = KNN(**knn_kwargs)
        cf_detector = OFI(rho=rho, r_hat=r_hat, ART=ART)
        cf_detector.OI.ARTMAP.module_a.set_data_bounds(
            np.zeros((X_train.shape[1],)),
            np.ones((X_train.shape[1],))
        )

        tpr_trace: List[List[float]] = []
        oi_trace: List[float] = []
        ofi_trace: List[Tuple[float, float]] = []
        cluster_forgetting_trace: List[List[float]] = []
        cluster_overshadowing_trace: List[List[float]] = []

        n_batches = len(y_train) // batch_size
        for i in range(n_batches):
            x_b = X_train[i * batch_size:(i + 1) * batch_size]
            y_b = y_train[i * batch_size:(i + 1) * batch_size]

            # 1) update memory and refit KNN
            X_mem.append(x_b)
            y_mem.append(y_b)
            X_seen = np.vstack(X_mem)
            y_seen = np.concatenate(y_mem)
            knn.fit(X_seen, y_seen)

            # 2) activations and predictions
            scores = knn.activation(X_test)
            y_pred = knn.predict(X_test)

            # 3) TPR per class
            tprs = []
            for cls in range(n_true):
                mask = (y_test == cls)
                tp = np.sum(y_pred[mask] == cls)
                fn = np.sum(y_pred[mask] != cls)
                tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            tpr_trace.append(tprs)

            # 4) global OFI update
            x_b_prep = cf_detector.OI.ARTMAP.module_a.prepare_data(x_b)
            O, F = cf_detector.add_batch(
                X_train=x_b_prep,
                y_train=y_b,
                y_pred_eval=y_pred,
                y_true_eval=y_test,
                y_scores_eval=scores
            )
            ofi_trace.append((O, F))

            # 5) Overlap Index
            oi_trace.append(cf_detector.OI.index)

            # 6) Per-cluster O and F (padded)
            forgetting_row = []
            overshadowing_row = []
            for cls in range(n_true):
                st = cf_detector.cluster_indices.get(cls, {})
                forgetting_row.append(st.get("forgetting", 0.0))
                overshadowing_row.append(st.get("overshadowing", 0.0))
            cluster_forgetting_trace.append(forgetting_row)
            cluster_overshadowing_trace.append(overshadowing_row)

        return (
            tpr_trace,
            oi_trace,
            ofi_trace,
            cluster_forgetting_trace,
            cluster_overshadowing_trace
        )

    traces_tpr: Dict[str, np.ndarray] = {}
    traces_oi: Dict[str, np.ndarray] = {}
    traces_state: Dict[str, np.ndarray] = {}
    traces_forg: Dict[str, np.ndarray] = {}
    traces_over: Dict[str, np.ndarray] = {}

    for ov, od in ((False, False), (False, True), (True, False), (True, True)):
        X_tr, y_tr, X_te, y_te = generate_synthetic_blobs(
            n_samples=3600,
            test_size=600,
            overlap=ov,
            ordered=od,
            random_state=SEED
        )
        cond = ("Overlapped, " if ov else "Separated, ") + (
            "Ordered" if od else "Shuffled")
        print(f"→ running condition {cond}")

        (tpr, oi_idx, ofi_states,
         cluster_f, cluster_o) = run_condition(
            X_tr, y_tr, X_te, y_te,
            batch_size=20,
            rho=0.95,
            r_hat=0.1,
            ART="Fuzzy",
            knn_kwargs={"n_neighbors": 1}
        )

        traces_tpr[cond] = np.array(tpr, dtype=float)
        traces_oi[cond] = np.array(oi_idx, dtype=float)
        traces_state[cond] = np.array(ofi_states, dtype=float)
        traces_forg[cond] = np.array(cluster_f, dtype=float)
        traces_over[cond] = np.array(cluster_o, dtype=float)

    path = "results_data/OFI/synthetic/ofi_knn_traces_synthetic_data.npz"
    make_dirs(path)
    np.savez(
        path,
        ov_od_tpr=traces_tpr["Overlapped, Ordered"],
        ov_sh_tpr=traces_tpr["Overlapped, Shuffled"],
        sp_od_tpr=traces_tpr["Separated, Ordered"],
        sp_sh_tpr=traces_tpr["Separated, Shuffled"],
        ov_od_int=traces_oi["Overlapped, Ordered"],
        ov_sh_int=traces_oi["Overlapped, Shuffled"],
        sp_od_int=traces_oi["Separated, Ordered"],
        sp_sh_int=traces_oi["Separated, Shuffled"],
        ov_od_states=traces_state["Overlapped, Ordered"],
        ov_sh_states=traces_state["Overlapped, Shuffled"],
        sp_od_states=traces_state["Separated, Ordered"],
        sp_sh_states=traces_state["Separated, Shuffled"],
        ov_od_forg=traces_forg["Overlapped, Ordered"],
        ov_sh_forg=traces_forg["Overlapped, Shuffled"],
        sp_od_forg=traces_forg["Separated, Ordered"],
        sp_sh_forg=traces_forg["Separated, Shuffled"],
        ov_od_over=traces_over["Overlapped, Ordered"],
        ov_sh_over=traces_over["Overlapped, Shuffled"],
        sp_od_over=traces_over["Separated, Ordered"],
        sp_sh_over=traces_over["Separated, Shuffled"],
    )

    print("Saved KNN-based OFI traces with per-cluster and global metrics.")


if __name__ == "__main__":
    experiment_ofi_synthetic_data_cnn()
    experiment_ofi_synthetic_data_knn()
