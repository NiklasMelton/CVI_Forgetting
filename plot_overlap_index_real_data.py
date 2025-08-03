import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import ListedColormap
import numpy as np
import pickle

from sklearn.manifold import TSNE
from scipy.spatial import procrustes

from tqdm import tqdm

from common import make_dirs



def plot_oi_mnist_combined_indices():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11
    })


    # Load data
    path = "results_data/overlap_index/real/mnist_oi.pickle"
    data = pickle.load(open(path, "rb"))
    oi_conv1 = data["oi_conv1_pooled"]
    cn_conv1 = data["cn_conv1_pooled"]
    oi_conv2 = data["oi_conv2_pooled"]
    cn_conv2 = data["cn_conv2_pooled"]
    oi_fc1 = data["oi_fc1"]
    cn_fc1 = data["cn_fc1"]
    oi_val_raw = data["oi_raw"]
    cn_val_raw = data["cn_raw"]

    val_accuracy_history = data["val_accuracy_history"]

    sil_conv1 = data["sil_conv1_pooled"]
    sil_conv2 = data["sil_conv2_pooled"]
    sil_fc1 = data["sil_fc1"]
    sil_val_raw = data["sil_raw"]

    batches_oi = 5 * np.arange(len(oi_conv1))
    batches_acc = np.arange(len(val_accuracy_history))

    # Create broken y-axis with two subplots
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1]}
    )

    # Plot same curves on both axes
    for ax in (ax_top, ax_bottom):
        # OI - blue shades
        ax.plot(batches_oi, oi_conv1, label='OI Pool1', color='#0072B2')
        ax.plot(batches_oi, oi_conv2, label='OI Pool2', color='#479CD5')
        ax.plot(batches_oi, oi_fc1,  label='OI FC1',   color='#A1CBE4')

        # CONN - green shades
        ax.plot(batches_oi, cn_conv1, label='CONN Pool1', color='#009E73')
        ax.plot(batches_oi, cn_conv2, label='CONN Pool2', color='#53B69F')
        ax.plot(batches_oi, cn_fc1,  label='CONN FC1',   color='#A5D6C8')

        # SIL - vermilion shades
        ax.plot(batches_oi, sil_conv1, label='Sil Pool1', color='#D55E00')
        ax.plot(batches_oi, sil_conv2, label='Sil Pool2', color='#E88F4D')
        ax.plot(batches_oi, sil_fc1,  label='Sil FC1',   color='#F2B78A')

        ax.hlines(oi_val_raw, xmin=batches_oi[0], xmax=batches_oi[-1],
                  linestyle='--', label='OI raw data', color='#0072B2')
        ax.hlines(cn_val_raw, xmin=batches_oi[0], xmax=batches_oi[-1],
                  linestyle='--', label='CONN raw data', color='#009E73')
        ax.hlines(sil_val_raw, xmin=batches_oi[0], xmax=batches_oi[-1],
                  linestyle='--', label='Sil raw data', color='#D55E00')

        ax.grid(True)



    # Zoom ranges
    ax_top.set_ylim(0.75, 1.01)
    ax_bottom.set_ylim(-0.01, 0.4)

    # Hide spines between axes
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)  # no top ticks
    ax_bottom.xaxis.tick_bottom()

    # Diagonal lines to indicate broken axis
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-0.01, +0.01), (-0.02, +0.02), **kwargs)
    ax_top.plot((0.99, 1.01), (-0.02, +0.02), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-0.01, +0.01), (1 - 0.02, 1 + 0.02), **kwargs)
    ax_bottom.plot((0.99, 1.01), (1 - 0.02, 1 + 0.02), **kwargs)

    # Accuracy on a secondary axis
    ax_acc = ax_top.twinx()
    ax_acc.plot(batches_acc, val_accuracy_history, color='gray', alpha=1.0, label='val accuracy', linestyle=":")
    ax_acc.set_ylim(0.75, 1.01)
    ax_acc.set_ylabel('Validation Accuracy')


    # Labeling
    ax_bottom.set_xlabel("Batch Index")
    ax_top.set_ylabel("Index Values")
    ax_bottom.set_ylabel("Index Values")
    fig.suptitle('Overlap and CONN Index for MNIST at Each Feature Level')

    # Legends
    # Combine and split legend handles
    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()

    # Split: 12 index lines and 1 val accuracy line
    index_lines = lines1
    index_labels = labels1

    accuracy_line = lines2[0]
    accuracy_label = labels2[0]

    # Add a dummy handle for spacing if needed
    empty_handle = plt.Line2D([], [], linestyle='none', label='')

    # Build combined legend: 12 + spacer + accuracy in its own column
    fig.legend(
        index_lines + [empty_handle] + [accuracy_line],
        index_labels + [''] + [accuracy_label],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,  # 5 for index + 1 for accuracy
        frameon=True,
        title='Indices'
    )



    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    path = "figures/overlap_index/real/mnist_oi_accuracy_revised.png"
    make_dirs(path)
    plt.savefig(path, bbox_inches='tight')

def plot_cnn_architecture():

    # Define layers with labels and sizes
    layers = [
        ("Input\n1×28×28", "#cce5ff"),
        ("Conv1\n16 @ 5×5", "#cce5ff"),
        ("Pool1\n16×14×14", "#cce5ff"),
        ("Conv2\n32 @ 5×5", "#cce5ff"),
        ("Pool2\n32×7×7", "#cce5ff"),
        ("FC1\n128 units", "#cce5ff"),
        ("FC2\n10 units", "#cce5ff"),
    ]

    fig, ax = plt.subplots(figsize=(3, 10))

    # Layout parameters
    box_width = 2.2
    box_height = 0.8
    v_spacing = 1.5

    # Compute positions for each layer
    positions = {}
    for i, (label, _) in enumerate(layers):
        x = 0
        y = -i * v_spacing
        positions[label] = (x, y)

    # Draw boxes
    for label, color in layers:
        x, y = positions[label]
        rect = FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            edgecolor="black",
            facecolor=color,
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, weight='bold')

    # Draw arrows between boxes
    for i in range(len(layers) - 1):
        label_from, _ = layers[i]
        label_to, _ = layers[i + 1]
        x0, y0 = positions[label_from]
        x1, y1 = positions[label_to]
        arrow = FancyArrowPatch(
            (x0, y0 - box_height / 2),
            (x1, y1 + box_height / 2),
            arrowstyle='-|>',
            mutation_scale=20,
            linewidth=1.5,
            color='black',
            shrinkA=0, shrinkB=0
        )
        ax.add_patch(arrow)

    # Final styling
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-len(layers) * v_spacing + 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("CNN Architecture\nfor MNIST", fontsize=20)
    plt.tight_layout()

    path = "figures/overlap_index/real/cnn_arch.png"
    make_dirs(path)
    plt.savefig(path, bbox_inches='tight')

def plot_embeddings():

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'legend.title_fontsize': 11
    })

    # Color Universal Design (CUD) 10-color palette
    CUD_COLORS = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#999999",  # gray
        "#000000",  # black
        "#A6761D"  # brown (from ColorBrewer extension)
    ]

    custom_cmap = ListedColormap(CUD_COLORS)

    # --- Brute-force mirrored Procrustes alignment ---
    def best_procrustes_alignment(X, X_ref):
        variants = [
            X,
            X * np.array([-1, 1]),  # flip X
            X * np.array([1, -1]),  # flip Y
            X * -1  # flip both
        ]

        best_X_aligned = None
        best_disparity = float("inf")

        for variant in variants:
            _, mtx2, disparity = procrustes(X_ref, variant)
            if disparity < best_disparity:
                best_disparity = disparity
                best_X_aligned = mtx2

        return best_X_aligned

    # --- Load the data ---
    path = "results_data/overlap_index/real/mnist_embeddings.pickle"
    data = pickle.load(open(path, "rb"))

    # Define keys and number of entries
    keys = ["X1", "X2", "X3"]
    n_rows = len(keys)
    n_cols = len(data)

    # Define readable labels
    key_mapping = {
        "X1": "Pool1",
        "X2": "Pool2",
        "X3": "FC1",
    }
    entry_mapping = {
        0: "Before Training",
        1: "Mid Training",
        2: "After Training"
    }

    # Set up the matplotlib figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    # Loop through keys and data entries
    for i, key in tqdm(enumerate(keys), desc="keys", total=n_rows):  # row index
        ref_layout = None  # to hold the reference layout for this key
        for j in tqdm(range(n_cols), desc="entries", leave=False):  # column index
            y = data[j]["y"]
            X = data[j][key]

            # Reduce dimensionality with t-SNE
            tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto',
                        max_iter=1000, random_state=0)
            X_tsne = tsne.fit_transform(X)

            # Align to first layout for this key
            if j == 0:
                ref_layout = X_tsne  # anchor layout
                layout_aligned = X_tsne
            else:
                layout_aligned = best_procrustes_alignment(X_tsne, ref_layout)

            # Plot
            ax = axes[i, j]
            scatter = ax.scatter(layout_aligned[:, 0], layout_aligned[:, 1], alpha=0.6,
                                 s=5, c=y, cmap=custom_cmap)
            title = f"{key_mapping[key]}, {entry_mapping[j]}"
            ax.set_title(title)
            ax.set_xlabel("tSNE-1")
            ax.set_ylabel("tSNE-2")

    # Create legend handles for class colors (0–9)
    legend_elements = [Patch(color=CUD_COLORS[i], label=f"Class {i}") for i in
                       range(10)]

    # Add legend to the top center of the figure
    fig.legend(handles=legend_elements, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03))

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # adjust layout to make space for legend
    path = "figures/overlap_index/real/mnist_embeddings.png"
    make_dirs(path)
    fig.savefig(path, bbox_inches='tight')


if __name__ == "__main__":
    plot_oi_mnist_combined_indices()
    plot_cnn_architecture()
    plot_embeddings()

