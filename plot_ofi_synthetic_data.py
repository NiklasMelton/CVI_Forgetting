import numpy as np
import matplotlib.pyplot as plt
from common import make_dirs

def plot_cnn_traces():

    # Load data
    path = "results_data/OFI/synthetic/ofi_cnn_traces_synthetic_data.npz"
    data = np.load(path)

    conditions = ['Separated, Ordered', 'Overlapped, Ordered', 'Separated, Shuffled',
                  "Overlapped, Shuffled"]
    class_labels = ['Class 0', 'Class 1', 'Class 2']

    prefix = {'Separated': 'sp', 'Overlapped': 'ov', 'Ordered': 'od', 'Shuffled': 'sh'}

    # Okabe-Ito palette
    okabe_ito_colors = [
        "#E69F00",  # Class 0 TPR
        "#56B4E9",  # Class 1 TPR
        "#009E73",  # Class 2 TPR
        "#0072B2",  # Overlap Index
        "#D55E00",  # Overshadowing (O)
        "#000000",  # Forgetting (F)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    # Store handles for legend only once
    legend_handles = []

    for ax_idx, (ax, cond) in enumerate(zip(axes, conditions)):
        p1, p2 = cond.split(", ")
        p = prefix[p1] + "_" + prefix[p2]
        tpr = data[f'{p}_tpr']
        oi = data[f'{p}_int']
        ofr = data[f'{p}_states']

        O, F = ofr[:, 0], ofr[:, 1]

        # Plot TPRs
        for i, lbl in enumerate(class_labels):
            line, = ax.plot(tpr[:, i], label=lbl + " TPR", color=okabe_ito_colors[i])
            if ax_idx == 0:  # only collect legend items once
                legend_handles.append(line)

        # Plot Overlap Index
        line, = ax.plot(oi, label='Overlap Index', linestyle='--',
                        color=okabe_ito_colors[3], linewidth=2)
        if ax_idx == 0:
            legend_handles.append(line)

        # Plot Overshadowing
        line, = ax.plot(O, label='Overshadowing (O)', linestyle='-.',
                        color=okabe_ito_colors[4], linewidth=2)
        if ax_idx == 0:
            legend_handles.append(line)

        # Plot Forgetting
        line, = ax.plot(F, label='Forgetting    (F)', linestyle=':',
                        color=okabe_ito_colors[5], linewidth=2)
        if ax_idx == 0:
            legend_handles.append(line)

        ax.set_title(cond)
        ax.set_ylabel('Metric Value')
        ax.grid(True)

    # Axis labels
    axes[2].set_xlabel('Batch Index')
    axes[3].set_xlabel('Batch Index')

    # Shared legend at top center
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # make space for legend
    plt.subplots_adjust(hspace=0.05)
    path = "figures/OFI/synthetic/cnn_traces.png"
    make_dirs(path)
    plt.savefig(path,
                bbox_inches='tight')


def plot_knn_traces():

    # Load data
    path = "results_data/OFI/synthetic/ofi_knn_traces_synthetic_data.npz"
    data = np.load(path)

    conditions = ['Separated, Ordered', 'Overlapped, Ordered', 'Separated, Shuffled',
                  "Overlapped, Shuffled"]
    class_labels = ['Class 0', 'Class 1', 'Class 2']

    prefix = {'Separated': 'sp', 'Overlapped': 'ov', 'Ordered': 'od', 'Shuffled': 'sh'}

    # Limited Okabe-Ito color palette (6 colors)
    okabe_ito_colors = [
        "#E69F00",  # Class 0 TPR
        "#56B4E9",  # Class 1 TPR
        "#009E73",  # Class 2 TPR
        "#0072B2",  # Overlap Index
        "#D55E00",  # Overshadowing (O)
        "#000000",  # Forgetting (F)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()

    legend_handles = []

    for ax_idx, (ax, cond) in enumerate(zip(axes, conditions)):
        p1, p2 = cond.split(", ")
        p = prefix[p1] + "_" + prefix[p2]
        tpr = data[f'{p}_tpr']  # (n_batches, 3)
        oi = data[f'{p}_int']  # (n_batches,)
        ofr = data[f'{p}_states']  # (n_batches, 2)

        O, F = ofr[:, 0], ofr[:, 1]

        # TPR per class
        for i, lbl in enumerate(class_labels):
            line, = ax.plot(tpr[:, i], label=lbl + " TPR", color=okabe_ito_colors[i])
            if ax_idx == 0:
                legend_handles.append(line)

        # Overlap Index
        line, = ax.plot(oi, label='Overlap Index', linestyle='--',
                        color=okabe_ito_colors[3], linewidth=2)
        if ax_idx == 0:
            legend_handles.append(line)

        # Overshadowing
        line, = ax.plot(O, label='Overshadowing (O)', linestyle='-.',
                        color=okabe_ito_colors[4], linewidth=2)
        if ax_idx == 0:
            legend_handles.append(line)

        # Forgetting
        line, = ax.plot(F, label='Forgetting    (F)', linestyle=':',
                        color=okabe_ito_colors[5], linewidth=2)
        if ax_idx == 0:
            legend_handles.append(line)

        ax.set_title(cond)
        ax.set_ylabel("Metric Value")
        ax.grid(True)

    axes[2].set_xlabel("Batch Index")
    axes[3].set_xlabel("Batch Index")

    # Shared legend at the top
    fig.legend(handles=legend_handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    path = "figures/OFI/synthetic/knn_traces.png"
    make_dirs(path)
    plt.savefig(path,
                bbox_inches='tight')

if __name__ == "__main__":
    plot_cnn_traces()
    plot_knn_traces()