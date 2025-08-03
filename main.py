from experiment_overlap_index_synthetic_data import experiment_cross, \
    experiment_circle, experiment_ring, experiment_bars
from experiment_ofi_synthetic_data import experiment_ofi_synthetic_data_cnn, \
    experiment_ofi_synthetic_data_knn
from experiment_overlap_index_real_data import experiment_oi_cnn
from experiment_ofi_mnist import experiment_ofi_mnist_cnn, experiment_ofi_mnist_knn
from experiment_ofi_cifar import experiment_ofi_cifar_cnn, experiment_ofi_cifar_knn

from plot_overlap_index_synthetic import plot_oi_synthetic_combined_indices, \
    plot_combined_exemplars
from plot_ofi_synthetic_data import plot_ofi_synthetic_knn_traces, plot_ofi_synthetic_cnn_traces
from plot_overlap_index_real_data import plot_oi_mnist_combined_indices, \
    plot_cnn_architecture, plot_embeddings
from plot_ofi_mnist import plot_ofi_mnist
from plot_ofi_cifar import plot_ofi_cifar


if __name__ == "__main__":
    print("Overlap Index Synthetic Experiments")
    experiment_circle()
    experiment_ring()
    experiment_bars()
    experiment_cross()
    plot_oi_synthetic_combined_indices()
    plot_combined_exemplars()

    print("OFI Synthetic Experiments")
    experiment_ofi_synthetic_data_cnn()
    experiment_ofi_synthetic_data_knn()
    plot_ofi_synthetic_cnn_traces()
    plot_ofi_synthetic_knn_traces()

    print("Overlap Index MNIST Experiments")
    experiment_oi_cnn()
    plot_oi_mnist_combined_indices()
    plot_cnn_architecture()
    plot_embeddings()

    print("OFI MNIST Experiments")
    experiment_ofi_mnist_cnn()
    experiment_ofi_mnist_knn()
    plot_ofi_mnist()

    print("OFI CIFAR Experiments")
    experiment_ofi_cifar_cnn()
    experiment_ofi_cifar_knn()
    plot_ofi_cifar()