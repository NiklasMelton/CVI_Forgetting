# === PLOTTING IMPORTS ===
from plot_overlap_index_synthetic import (
    plot_oi_synthetic_combined_indices,
    plot_combined_exemplars,
)
from plot_ofi_synthetic_data import (
    plot_ofi_synthetic_cnn_traces,
    plot_ofi_synthetic_knn_traces,
)
from plot_overlap_index_real_data import (
    plot_oi_mnist_combined_indices,
    plot_cnn_architecture,
    plot_embeddings,
)
from plot_ofi_mnist import plot_ofi_mnist
from plot_ofi_cifar import plot_ofi_cifar


def main():

    plot_oi_synthetic_combined_indices()
    plot_combined_exemplars()

    plot_ofi_synthetic_cnn_traces()
    plot_ofi_synthetic_knn_traces()

    plot_oi_mnist_combined_indices()
    plot_cnn_architecture()
    plot_embeddings()

    plot_ofi_mnist()
    plot_ofi_cifar()

if __name__ == "__main__":
    main()