# === PLOTTING IMPORTS ===
from plot_overlap_index_synthetic import (
    plot_oi_synthetic_combined_indices_all_seeds,
    plot_combined_exemplars,
)
from plot_ofi_synthetic_data import (
    plot_ofi_synthetic_cnn_traces_all_seeds,
    plot_ofi_synthetic_knn_traces_all_seeds,
)
from plot_overlap_index_real_data import (
    plot_oi_mnist_combined_indices_all_seeds,
    plot_cnn_architecture,
    plot_embeddings,
)
from plot_ofi_mnist import plot_ofi_mnist_all_seeds
from plot_ofi_cifar import plot_ofi_cifar_all_seeds


def main():

    plot_oi_synthetic_combined_indices_all_seeds()
    plot_combined_exemplars()

    plot_ofi_synthetic_cnn_traces_all_seeds()
    plot_ofi_synthetic_knn_traces_all_seeds()

    plot_oi_mnist_combined_indices_all_seeds()
    plot_cnn_architecture()
    plot_embeddings()

    plot_ofi_mnist_all_seeds()
    plot_ofi_cifar_all_seeds()

if __name__ == "__main__":
    main()