import numpy as np
import matplotlib.pyplot as plt
from common import make_dirs, find_seeded, find_all_seeded, find_lowest_seeded

def plot_ofi_cifar():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12
    })

    # Load data
    cnn_path = "results_data/OFI/real/ofi_traces_cifar_cnn.npz"
    knn_path = "results_data/OFI/real/ofi_traces_cifar_knn.npz"
    cnn_path, knn_path = find_seeded([cnn_path, knn_path])
    cnn_data = np.load(cnn_path, allow_pickle=True)
    knn_data = np.load(knn_path, allow_pickle=True)

    methods = ["CNN", "KNN"]
    datasets = [cnn_data, knn_data]
    orders = ["ordered", "shuffled"]
    colors = [
        "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
        "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100",
        "#6699CC", "#4477AA", "#AA4466", "#000000"
    ]
    class_labels = [f"Class {i} TPR" for i in range(10)]

    # === Simplified Mosaic Layout (No Broken Axis) ===
    fig = plt.figure(figsize=(14, 8))
    mosaic = [
        ["A", "B"],   # Ordered
        [".", "."],   # Spacer
        ["C", "D"],   # Shuffled (Single subplot)
    ]
    ax_dict = fig.subplot_mosaic(mosaic, height_ratios=[1.5, 0.1, 1.5])
    plt.subplots_adjust(hspace=0.1, wspace=0.25, top=0.84, bottom=0.07)  # reserve top space

    # Legend handles
    tpr_lines = []
    metric_lines = []
    metric_labels = []

    # === Plotting ===
    for col, (method, data) in enumerate(zip(methods, datasets)):
        for row, order in enumerate(orders):
            prefix = order
            tpr = data[f"{prefix}_tpr"]
            acc = data[f"{prefix}_accuracy"]
            ofi = data[f"{prefix}_states"]
            x = np.arange(len(acc))

            ax = ax_dict[["A", "B", "C", "D"][row * 2 + col]]
            ax2 = ax.twinx()

            # Plot TPRs
            for i in range(10):
                line, = ax.plot(x, tpr[:, i], color=colors[i], linewidth=1, alpha=0.7)
                if row == 0 and col == 0:
                    tpr_lines.append(line)

            # Accuracy and OFI
            acc_line, = ax.plot(x, acc, color=colors[-1], linewidth=2)
            o_line, = ax2.plot(x, ofi[:, 0], linestyle="--", color=colors[-2], linewidth=2)
            f_line, = ax2.plot(x, ofi[:, 1], linestyle="-.", color=colors[-3], linewidth=2)

            # Y-axis limits
            ax.set_ylim(0.0, 1.05)
            ax2.set_ylim(0.0, 1.05)

            # Labels
            if col == 0:
                ax.set_ylabel("TPR / Accuracy")
            if col == 1:
                ax2.set_ylabel("O / F")

            # Titles and x-labels
            ax.set_title(f"{method} - {order.capitalize()}", fontsize=12)
            if row == 1:
                ax.set_xlabel("Batch Index")

            # Legend setup
            if row == 0 and col == 0:
                metric_lines = [acc_line, o_line, f_line]
                metric_labels = ["Accuracy", "Overshadowing", "Forgetting"]

    # === Legends (Side-by-side, Centered, Equal Height) ===
    legend_y = 1.0
    legend_x_offset = 0.375
    fig.legend(tpr_lines, class_labels,
               loc="upper center", bbox_to_anchor=(legend_x_offset, legend_y),
               fontsize=12, title="Class True-Positive Rates", ncol=5, frameon=True)
    fig.legend(metric_lines, metric_labels,
               loc="upper center", bbox_to_anchor=(legend_x_offset+0.44, legend_y),
               fontsize=12, title="Global Metrics", ncol=2, frameon=True)

    path = "figures/OFI/real/cifar_ofi_cnn_knn.png"
    make_dirs(path)
    plt.savefig(path, bbox_inches='tight')



def plot_ofi_cifar_all_seeds():
    # 1) collect all seed files
    cnn_base = "results_data/OFI/real/ofi_traces_cifar_cnn.npz"
    knn_base = "results_data/OFI/real/ofi_traces_cifar_knn.npz"
    cnn_files = find_all_seeded(cnn_base)
    knn_files = find_all_seeded(knn_base)

    # pick lowest-seed file for each
    cnn_lowest_fp = find_lowest_seeded(cnn_files)
    knn_lowest_fp = find_lowest_seeded(knn_files)

    # load
    cnn_lowest = np.load(cnn_lowest_fp, allow_pickle=True)
    knn_lowest = np.load(knn_lowest_fp, allow_pickle=True)
    cnn_seeds = [np.load(fp, allow_pickle=True) for fp in cnn_files]
    knn_seeds = [np.load(fp, allow_pickle=True) for fp in knn_files]

    methods = ["CNN", "KNN"]
    lowest_data = [cnn_lowest, knn_lowest]
    all_seeds   = [cnn_seeds,  knn_seeds]

    orders = ["ordered", "shuffled"]
    colors = [
        "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
        "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100",
        "#6699CC", "#4477AA", "#AA4466", "#000000"
    ]
    class_labels = [f"Class {i} TPR" for i in range(10)]
    metric_labels = ["Accuracy", "Overshadowing", "Forgetting"]

    # layout
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12
    })
    fig = plt.figure(figsize=(14, 8))
    mosaic = [
        ["A", "B"],
        [".", "."],
        ["C", "D"],
    ]
    axd = fig.subplot_mosaic(mosaic, height_ratios=[1.5, 0.1, 1.5])
    plt.subplots_adjust(hspace=0.1, wspace=0.25, top=0.84, bottom=0.07)

    tpr_lines = []
    metric_handles = []

    for col, (method, low, seeds) in enumerate(zip(methods, lowest_data, all_seeds)):
        for row, order in enumerate(orders):
            ax = axd[["A","B","C","D"][row*2 + col]]
            ax2 = ax.twinx()
            prefix = order

            # --- TPR from lowest-seed only ---
            tpr = low[f"{prefix}_tpr"]       # shape (n_batches, 10)
            x = np.arange(tpr.shape[0])
            for i in range(10):
                ln, = ax.plot(x, tpr[:, i], color=colors[i], linewidth=1, alpha=0.7)
                if row==0 and col==0:
                    tpr_lines.append(ln)

            # --- Accuracy, O, F with CI ---
            # stack acc: (n_seeds, batches)
            acc_stack = np.vstack([sd[f"{prefix}_accuracy"] for sd in seeds])
            oi_stack  = np.stack([sd[f"{prefix}_states"][:,0] for sd in seeds], axis=0)
            of_stack  = np.stack([sd[f"{prefix}_states"][:,1] for sd in seeds], axis=0)

            def plot_ci(stack, style, clr, ax_target):
                mean = stack.mean(axis=0)
                sem  = stack.std(ddof=1, axis=0)/np.sqrt(stack.shape[0])
                ci95 = 1.96*sem
                ln, = ax_target.plot(x, mean, style, color=clr, linewidth=2)
                ax_target.fill_between(x, mean-ci95, mean+ci95, color=clr, alpha=0.2)
                return ln

            ln_acc = plot_ci(acc_stack, "-", colors[-1], ax)
            ln_o   = plot_ci(oi_stack, "--", colors[-2], ax2)
            ln_f   = plot_ci(of_stack, "-.", colors[-3], ax2)
            if row==0 and col==0:
                metric_handles = [ln_acc, ln_o, ln_f]

            ax.set_ylim(0, 1.05)
            ax2.set_ylim(0, 1.05)

            if col==0:
                ax.set_ylabel("TPR / Accuracy")
            if col==1:
                ax2.set_ylabel("O / F")

            ax.set_title(f"{method} – {order.capitalize()}", fontsize=12)
            if row==1:
                ax.set_xlabel("Batch Index")

            ax.grid(True)

    # legends
    fig.legend(tpr_lines, class_labels,
               loc="upper center", bbox_to_anchor=(0.375, 1.0),
               title="Class TPRs", ncol=5, frameon=True)
    fig.legend(metric_handles, metric_labels,
               loc="upper center", bbox_to_anchor=(0.815, 1.0),
               title="Global Metrics", ncol=3, frameon=True)

    # save
    out = "figures/OFI/real/cifar_ofi_cnn_knn_all_seeds.png"
    make_dirs(out)
    plt.savefig(out, bbox_inches='tight')


if __name__ == "__main__":
    plot_ofi_cifar()