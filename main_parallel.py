import os
import random
import multiprocessing as mp

# === USER PARAMETERS ===
NUM_SEEDS    = 20        # how many different random seeds
NUM_JOBS     = 20        # must match SBATCH --ntasks
INITIAL_SEED = 12345     # for reproducible seed list generation

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

_SYNTHETIC_FUNCS = {
    experiment_circle,
    experiment_ring,
    experiment_bars,
    experiment_cross,
}

def _init_worker():
    """Pin each worker’s internal BLAS/OpenMP threads to SLURM_CPUS_PER_TASK."""
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    os.environ["OMP_NUM_THREADS"] = str(cpus)
    os.environ["MKL_NUM_THREADS"] = str(cpus)

    # PyTorch
    import torch
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(cpus)

    # TensorFlow
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(cpus)
    tf.config.threading.set_inter_op_parallelism_threads(cpus)

def _run_experiment(func, seed):
    """Wrapper: always pass seed; for synthetic funcs if NUM_SEEDS > 1 force n_seeds=1."""
    kwargs = {"seed": seed}
    if func in _SYNTHETIC_FUNCS and NUM_SEEDS != 1:
        kwargs["n_seeds"] = 1
    func(**kwargs)

def main():
    # 1) safe start for TF
    mp.set_start_method("spawn", force=True)

    # 2) reproducible list of seeds
    rng = random.Random(INITIAL_SEED)
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(NUM_SEEDS)]

    # 3) all experiments (unchanged)
    experiment_funcs = [
        experiment_circle,
        experiment_ring,
        experiment_bars,
        experiment_cross,
        experiment_ofi_synthetic_data_cnn,
        experiment_ofi_synthetic_data_knn,
        experiment_oi_cnn,
        experiment_ofi_mnist_cnn,
        experiment_ofi_mnist_knn,
        experiment_ofi_cifar_cnn,
        experiment_ofi_cifar_knn,
    ]

    # 4) dispatch: 20 workers, each with 3 threads internally
    with mp.Pool(processes=NUM_JOBS, initializer=_init_worker) as pool:
        for func in experiment_funcs:
            for seed in seeds:
                pool.apply_async(_run_experiment, args=(func, seed))
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
