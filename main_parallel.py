import random
import multiprocessing

# === USER PARAMETERS ===
NUM_SEEDS    = 20       # how many different random seeds to try
NUM_JOBS     = 20        # how many worker processes to spawn
INITIAL_SEED = 12345    # for reproducible seed list generation

# === EXPERIMENT IMPORTS ===
from experiment_overlap_index_synthetic_data import (
    experiment_cross,
    experiment_circle,
    experiment_ring,
    experiment_bars,
)
from experiment_ofi_synthetic_data import (
    experiment_ofi_synthetic_data_cnn,
    experiment_ofi_synthetic_data_knn,
)
from experiment_overlap_index_real_data import experiment_oi_cnn
from experiment_ofi_mnist import (
    experiment_ofi_mnist_cnn,
    experiment_ofi_mnist_knn,
)
from experiment_ofi_cifar import (
    experiment_ofi_cifar_cnn,
    experiment_ofi_cifar_knn,
)


# identify the “circle/ring/bars/cross” set
_SYNTHETIC_FUNCS = {
    experiment_circle,
    experiment_ring,
    experiment_bars,
    experiment_cross,
}

def _run_experiment(func, seed):
    """Wrapper: always pass seed; for the 4 synthetic funcs, if NUM_SEEDS>1 force n_seeds=1."""
    kwargs = {"seed": seed}
    if func in _SYNTHETIC_FUNCS and NUM_SEEDS != 1:
        kwargs["n_seeds"] = 1
    func(**kwargs)

def main():
    # 1) reproducible seed list
    rng = random.Random(INITIAL_SEED)
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(NUM_SEEDS)]

    # 2) all experiments
    experiment_funcs = [
        # experiment_circle,
        # experiment_ring,
        # experiment_bars,
        # experiment_cross,
        # experiment_ofi_synthetic_data_cnn,
        # experiment_ofi_synthetic_data_knn,
        experiment_oi_cnn,
        experiment_ofi_mnist_cnn,
        experiment_ofi_mnist_knn,
        experiment_ofi_cifar_cnn,
        experiment_ofi_cifar_knn,
    ]

    # 3) dispatch in an async pool
    with multiprocessing.Pool(processes=NUM_JOBS) as pool:
        for func in experiment_funcs:
            for seed in seeds:
                pool.apply_async(_run_experiment, args=(func, seed))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
