import numpy as np
import matplotlib.pyplot as plt
from common import make_dirs, find_seeded, find_all_seeded

def plot_ofi_synthetic_cnn_traces():

    # Load data
    path = "results_data/OFI/synthetic/ofi_cnn_traces_synthetic_data.npz"
    path = find_seeded(path)
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


def plot_ofi_synthetic_cnn_traces_all_seeds():
    # find all the seeded .npz paths
    base_path = "results_data/OFI/synthetic/ofi_cnn_traces_synthetic_data.npz"
    fnames = find_all_seeded(base_path)
    if not fnames:
        raise FileNotFoundError(f"No files matching seeds for {base_path}")

    # load every seed
    seed_data = [np.load(fp) for fp in fnames]
    n_seeds = len(seed_data)

    conditions = [
        'Separated, Ordered', 'Overlapped, Ordered',
        'Separated, Shuffled', 'Overlapped, Shuffled'
    ]
    class_labels = ['Class 0', 'Class 1', 'Class 2']
    prefix = {'Separated': 'sp', 'Overlapped': 'ov', 'Ordered': 'od', 'Shuffled': 'sh'}

    # Okabe–Ito palette
    colors = [
        "#E69F00",  # Class 0
        "#56B4E9",  # Class 1
        "#009E73",  # Class 2
        "#0072B2",  # Overlap Index
        "#D55E00",  # Overshadowing
        "#000000",  # Forgetting
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    legend_handles = []

    for idx, (ax, cond) in enumerate(zip(axes, conditions)):
        p1, p2 = cond.split(", ")
        key = f"{prefix[p1]}_{prefix[p2]}"

        # stack TPR: shape (n_seeds, n_batches, n_classes)
        tpr_stack = np.stack([sd[f"{key}_tpr"] for sd in seed_data], axis=0)
        # stack Overlap Index: (n_seeds, n_batches)
        oi_stack  = np.stack([sd[f"{key}_int"] for sd in seed_data], axis=0)
        # stack O/F states: (n_seeds, n_batches, 2)
        ofr_stack = np.stack([sd[f"{key}_states"] for sd in seed_data], axis=0)

        n_batches = tpr_stack.shape[1]
        x = np.arange(n_batches)

        # plot TPRs with CI
        for ci, lbl in enumerate(class_labels):
            arr = tpr_stack[:, :, ci]
            mean = arr.mean(axis=0)
            sem  = arr.std(ddof=1, axis=0) / np.sqrt(n_seeds)
            ci95 = 1.96 * sem

            ln = ax.plot(x, mean, label=f"{lbl} TPR", color=colors[ci])[0]
            ax.fill_between(x, mean - ci95, mean + ci95, color=colors[ci], alpha=0.2)

            if idx == 0:
                legend_handles.append(ln)

        # Overlap Index
        arr = oi_stack
        mean = arr.mean(axis=0)
        ci95 = 1.96 * (arr.std(ddof=1, axis=0) / np.sqrt(n_seeds))
        ln = ax.plot(x, mean, linestyle='--', linewidth=2, label='Overlap Index', color=colors[3])[0]
        ax.fill_between(x, mean - ci95, mean + ci95, color=colors[3], alpha=0.2)
        if idx == 0:
            legend_handles.append(ln)

        # Overshadowing (O)
        arr = ofr_stack[:, :, 0]
        mean = arr.mean(axis=0)
        ci95 = 1.96 * (arr.std(ddof=1, axis=0) / np.sqrt(n_seeds))
        ln = ax.plot(x, mean, linestyle='-.', linewidth=2, label='Overshadowing (O)', color=colors[4])[0]
        ax.fill_between(x, mean - ci95, mean + ci95, color=colors[4], alpha=0.2)
        if idx == 0:
            legend_handles.append(ln)

        # Forgetting (F)
        arr = ofr_stack[:, :, 1]
        mean = arr.mean(axis=0)
        ci95 = 1.96 * (arr.std(ddof=1, axis=0) / np.sqrt(n_seeds))
        ln = ax.plot(x, mean, linestyle=':', linewidth=2, label='Forgetting (F)', color=colors[5])[0]
        ax.fill_between(x, mean - ci95, mean + ci95, color=colors[5], alpha=0.2)
        if idx == 0:
            legend_handles.append(ln)

        ax.set_title(cond)
        ax.set_ylabel('Metric Value')
        ax.grid(True)

    # x‐axis label only on bottom row
    for ax in axes[2:]:
        ax.set_xlabel('Batch Index')

    # shared legend
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.05)

    out_path = "figures/OFI/synthetic/cnn_traces_all_seeds.png"
    make_dirs(out_path)
    plt.savefig(out_path, bbox_inches='tight')


def plot_ofi_synthetic_knn_traces():

    # Load data
    path = "results_data/OFI/synthetic/ofi_knn_traces_synthetic_data.npz"
    path = find_seeded(path)
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


def plot_ofi_synthetic_knn_traces_all_seeds():
    # 1) collect all seed files
    base_path = "results_data/OFI/synthetic/ofi_knn_traces_synthetic_data.npz"
    fnames = find_all_seeded(base_path)
    n_seeds = len(fnames)
    if n_seeds == 0:
        raise FileNotFoundError(f"No files matching seeds for {base_path}")

    # 2) load them all
    seed_data = [np.load(fp) for fp in fnames]

    conditions = [
        'Separated, Ordered', 'Overlapped, Ordered',
        'Separated, Shuffled', 'Overlapped, Shuffled'
    ]
    class_labels = ['Class 0', 'Class 1', 'Class 2']
    prefix = {'Separated': 'sp', 'Overlapped': 'ov', 'Ordered': 'od', 'Shuffled': 'sh'}

    # Okabe–Ito palette
    colors = [
        "#E69F00",  # Class 0
        "#56B4E9",  # Class 1
        "#009E73",  # Class 2
        "#0072B2",  # Overlap Index
        "#D55E00",  # Overshadowing
        "#000000",  # Forgetting
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    legend_handles = []

    for idx, (ax, cond) in enumerate(zip(axes, conditions)):
        p1, p2 = cond.split(", ")
        key = f"{prefix[p1]}_{prefix[p2]}"

        # stack shapes:
        #   tpr_stack: (n_seeds, n_batches, 3)
        #   oi_stack:  (n_seeds, n_batches)
        #   ofr_stack: (n_seeds, n_batches, 2)
        tpr_stack = np.stack([sd[f"{key}_tpr"] for sd in seed_data], axis=0)
        oi_stack  = np.stack([sd[f"{key}_int"] for sd in seed_data], axis=0)
        ofr_stack = np.stack([sd[f"{key}_states"] for sd in seed_data], axis=0)

        n_batches = tpr_stack.shape[1]
        x = np.arange(n_batches)

        # helper to plot mean+CI
        def plot_with_ci(arr, style, clr, lbl):
            mean = arr.mean(axis=0)
            sem  = arr.std(ddof=1, axis=0) / np.sqrt(n_seeds)
            ci95 = 1.96 * sem
            ln = ax.plot(x, mean, style, label=lbl, color=clr, linewidth=2)[0]
            ax.fill_between(x, mean - ci95, mean + ci95, color=clr, alpha=0.2)
            return ln

        # plot TPRs
        for ci, lbl in enumerate(class_labels):
            ln = plot_with_ci(
                tpr_stack[:, :, ci],
                style='-',
                clr=colors[ci],
                lbl=f"{lbl} TPR"
            )
            if idx == 0:
                legend_handles.append(ln)

        # Overlap Index
        ln = plot_with_ci(
            oi_stack,
            style='--',
            clr=colors[3],
            lbl='Overlap Index'
        )
        if idx == 0:
            legend_handles.append(ln)

        # Overshadowing (O)
        ln = plot_with_ci(
            ofr_stack[:, :, 0],
            style='-.',
            clr=colors[4],
            lbl='Overshadowing (O)'
        )
        if idx == 0:
            legend_handles.append(ln)

        # Forgetting (F)
        ln = plot_with_ci(
            ofr_stack[:, :, 1],
            style=':',
            clr=colors[5],
            lbl='Forgetting (F)'
        )
        if idx == 0:
            legend_handles.append(ln)

        ax.set_title(cond)
        ax.set_ylabel("Metric Value")
        ax.grid(True)

    # only bottom row gets x‐label
    for ax in axes[2:]:
        ax.set_xlabel("Batch Index")

    # shared legend at top
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    out_path = "figures/OFI/synthetic/knn_traces_all_seeds.png"
    make_dirs(out_path)
    plt.savefig(out_path, bbox_inches='tight')


if __name__ == "__main__":
    plot_ofi_synthetic_cnn_traces()
    plot_ofi_synthetic_knn_traces()