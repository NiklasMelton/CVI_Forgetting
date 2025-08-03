import numpy as np
import matplotlib.pyplot as plt
from common import make_dirs, find_seeded, find_all_seeded, find_lowest_seeded

def plot_ofi_mnist():
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
    cnn_path = "results_data/OFI/real/ofi_traces_mnist_cnn.npz"
    knn_path = "results_data/OFI/real/ofi_traces_mnist_knn.npz"
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

    # === Mosaic Layout with Spacer ===
    fig = plt.figure(figsize=(14, 10))
    mosaic = [
        ["A", "B"],   # Ordered
        [".", "."],   # Spacer
        ["C", "D"],   # Shuffled (Top)
        ["E", "F"]    # Shuffled (Bottom)
    ]
    ax_dict = fig.subplot_mosaic(mosaic, height_ratios=[1.5, 0.1, 0.7, 0.5])
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

            if row == 0:
                ax = ax_dict[["A", "B"][col]]
                ax2 = ax.twinx()
            else:
                ax_top = ax_dict[["C", "D"][col]]
                ax_bot = ax_dict[["E", "F"][col]]
                ax = ax_top
                ax2 = ax_top.twinx()
                ax_bot2 = ax_bot.twinx()

                # Broken axis setup
                ax_top.spines['bottom'].set_visible(False)
                ax_bot.spines['top'].set_visible(False)
                ax_top.tick_params(labeltop=False)
                ax_top.set_xticklabels([])
                ax_bot.xaxis.tick_bottom()

                # Diagonal break markers
                kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
                ax_top.plot((-0.01, +0.01), (-0.015, +0.015), **kwargs)
                ax_top.plot((0.99, 1.01), (-0.015, +0.015), **kwargs)
                kwargs.update(transform=ax_bot.transAxes)
                ax_bot.plot((-0.01, +0.01), (1 - 0.015, 1 + 0.015), **kwargs)
                ax_bot.plot((0.99, 1.01), (1 - 0.015, 1 + 0.015), **kwargs)

            # Plot TPRs
            for i in range(10):
                line, = ax.plot(x, tpr[:, i], color=colors[i], linewidth=1, alpha=0.7)
                if row == 0 and col == 0:
                    tpr_lines.append(line)

            # Accuracy and OFI
            acc_line, = ax.plot(x, acc, color=colors[-1], linewidth=2)
            o_line, = ax2.plot(x, ofi[:, 0], linestyle="--", color=colors[-2], linewidth=2)
            f_line, = ax2.plot(x, ofi[:, 1], linestyle="-.", color=colors[-3], linewidth=2)

            if row == 1:
                ax_bot.plot(x, acc, color=colors[-1], linewidth=2)
                ax_bot2.plot(x, ofi[:, 0], linestyle="--", color=colors[-2], linewidth=2)
                ax_bot2.plot(x, ofi[:, 1], linestyle="-.", color=colors[-3], linewidth=2)

                ax.set_ylim(0.6, 1.0)
                ax2.set_ylim(0.6, 1.0)
                ax_bot.set_ylim(0.0, 0.1)
                ax_bot2.set_ylim(0.0, 0.1)

                # Centered Y-labels at break
                # ax.text(-0.08, 0.5, "TPR / Accuracy", transform=ax.transAxes,
                #         rotation=90, va='center', ha='center', fontsize=10)
                # ax2.text(1.08, 0.5, "O / F", transform=ax2.transAxes,
                #          rotation=90, va='center', ha='center', fontsize=10)
                if col == 0:
                    ax.set_ylabel("TPR / Accuracy")
                if col == 1:
                    ax2.set_ylabel("O / F")
            else:
                ax.set_ylim(0, 1.05)
                ax2.set_ylim(0, 1.05)
                if col == 0:
                    ax.set_ylabel("TPR / Accuracy")
                if col == 1:
                    ax2.set_ylabel("O / F")

            # Titles and x-labels
            ax.set_title(f"{method} - {order.capitalize()}", fontsize=14)
            if row == 1:
                ax_bot.set_xlabel("Batch Index")
            elif row == 3:
                ax.set_xlabel("Batch Index")

            if row == 0 and col == 0:
                metric_lines = [acc_line, o_line, f_line]
                metric_labels = ["Accuracy", "Overshadowing", "Forgetting"]

    # === Legends (Side-by-side, Centered, Equal Height) ===
    legend_y = 0.965
    legend_x_offset = 0.375
    fig.legend(tpr_lines, class_labels,
               loc="upper center", bbox_to_anchor=(legend_x_offset, legend_y),
               fontsize=12, title="Class True-Positive Rates", ncol=5, frameon=True)
    fig.legend(metric_lines, metric_labels,
               loc="upper center", bbox_to_anchor=(legend_x_offset+0.44, legend_y),
               fontsize=12, title="Global Metrics", ncol=2, frameon=True)
    path = "figures/OFI/real/mnist_ofi_cnn_knn.png"
    make_dirs(path)
    plt.savefig(path, bbox_inches='tight')


def plot_ofi_mnist_all_seeds():
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12
    })

    # 1) find all seed files
    cnn_base = "results_data/OFI/real/ofi_traces_mnist_cnn.npz"
    knn_base = "results_data/OFI/real/ofi_traces_mnist_knn.npz"
    cnn_files = find_all_seeded(cnn_base)
    knn_files = find_all_seeded(knn_base)

    # 2) pick the lowest-seed file for TPR plotting
    cnn_lowest_fp = find_lowest_seeded(cnn_files)
    knn_lowest_fp = find_lowest_seeded(knn_files)

    # 3) load lowest-seed data + all seeds
    cnn_lowest = np.load(cnn_lowest_fp, allow_pickle=True)
    knn_lowest = np.load(knn_lowest_fp, allow_pickle=True)
    cnn_seeds  = [np.load(fp, allow_pickle=True) for fp in cnn_files]
    knn_seeds  = [np.load(fp, allow_pickle=True) for fp in knn_files]

    methods      = ["CNN", "KNN"]
    lowest_data  = [cnn_lowest, knn_lowest]
    all_seeds    = [cnn_seeds,  knn_seeds]

    orders       = ["ordered", "shuffled"]
    colors       = [
        "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
        "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100",
        "#6699CC", "#4477AA", "#AA4466", "#000000"
    ]
    class_labels    = [f"Class {i} TPR" for i in range(10)]
    metric_labels   = ["Accuracy", "Overshadowing", "Forgetting"]

    # helper for CI shading
    def plot_ci(stack, style, clr, ax):
        mean = stack.mean(axis=0)
        sem  = stack.std(ddof=1, axis=0) / np.sqrt(len(stack))
        ci95 = 1.96 * sem
        ln, = ax.plot(x, mean, style, color=clr, linewidth=2)
        ax.fill_between(x, mean-ci95, mean+ci95, color=clr, alpha=0.2)
        return ln

    # === Mosaic layout ===
    fig = plt.figure(figsize=(14, 10))
    mosaic = [
        ["A", "B"],
        [".", "."],
        ["C", "D"],
        ["E", "F"],
    ]
    axd = fig.subplot_mosaic(mosaic, height_ratios=[1.5, 0.1, 0.7, 0.5])
    plt.subplots_adjust(hspace=0.1, wspace=0.25, top=0.84, bottom=0.07)

    tpr_lines    = []
    metric_lines = []

    for col, (method, low, seeds) in enumerate(zip(methods, lowest_data, all_seeds)):
        for row, order in enumerate(orders):
            prefix = order
            # lowest seed for TPR
            tpr_low = low[f"{prefix}_tpr"]           # (n_batches, 10)
            # all seeds for metrics
            acc_stack = np.vstack([sd[f"{prefix}_accuracy"] for sd in seeds])        # (n_seeds, n_batches)
            ofi_stack = np.stack([sd[f"{prefix}_states"] for sd in seeds], axis=0)    # (n_seeds, n_batches, 2)
            O_stack = ofi_stack[:,:,0]
            F_stack = ofi_stack[:,:,1]

            n_batches = tpr_low.shape[0]
            x = np.arange(n_batches)

            if row == 0:
                ax  = axd[["A","B"][col]]
                ax2 = ax.twinx()

                # --- TPRs from lowest seed ---
                for i in range(10):
                    ln, = ax.plot(x, tpr_low[:, i], color=colors[i], linewidth=1, alpha=0.7)
                    if col == 0:
                        tpr_lines.append(ln)

                # --- Accuracy, O, F with CI ---
                acc_ln = plot_ci(acc_stack, "-",  colors[-1], ax)
                o_ln   = plot_ci(O_stack,    "--", colors[-2], ax2)
                f_ln   = plot_ci(F_stack,    "-.", colors[-3], ax2)
                if col == 0:
                    metric_lines = [acc_ln, o_ln, f_ln]

                # axes & labels
                ax.set_ylim(0, 1.05)
                ax2.set_ylim(0, 1.05)
                ax.set_ylabel("TPR / Accuracy")
                ax2.set_ylabel("O / F")

                ax.set_title(f"{method} – {order.capitalize()}", fontsize=14)

            else:
                # broken-axis pair
                ax_top = axd[["C","D"][col]]
                ax_bot = axd[["E","F"][col]]
                ax2     = ax_top.twinx()
                ax_bot2 = ax_bot.twinx()

                # hide spines, add break markers
                ax_top.spines['bottom'].set_visible(False)
                ax_bot.spines['top'].set_visible(False)
                ax_top.tick_params(labeltop=False)
                ax_top.set_xticklabels([])
                ax_bot.xaxis.tick_bottom()
                for (a, b) in [(-0.01, +0.01), (0.99, 1.01)]:
                    ax_top.plot((a, b), (-0.015, +0.015), transform=ax_top.transAxes, color='k', clip_on=False)
                    ax_bot.plot((a, b), (1-0.015, 1+0.015), transform=ax_bot.transAxes, color='k', clip_on=False)

                # --- TPRs from lowest seed ---
                for i in range(10):
                    ax_top.plot(x, tpr_low[:, i], color=colors[i], linewidth=1, alpha=0.7)

                # --- Accuracy CI on both subplots ---
                plot_ci(acc_stack, "-",  colors[-1], ax_top)
                plot_ci(acc_stack, "-",  colors[-1], ax_bot)

                # --- O and F CI on both subplots ---
                plot_ci(O_stack, "--", colors[-2], ax2)
                plot_ci(O_stack, "--", colors[-2], ax_bot2)
                plot_ci(F_stack, "-.", colors[-3], ax2)
                plot_ci(F_stack, "-.", colors[-3], ax_bot2)

                # adjust y-limits
                ax_top.set_ylim(0.6, 1.0)
                ax2.set_ylim(0.6, 1.0)
                ax_bot.set_ylim(0.0, 0.1)
                ax_bot2.set_ylim(0.0, 0.1)

                # labels & title
                if col == 0:
                    ax_top.set_ylabel("TPR / Accuracy")
                else:
                    ax_top.set_ylabel("O / F")
                ax_top.set_title(f"{method} – {order.capitalize()}", fontsize=14)
                ax_bot.set_xlabel("Batch Index")

    # === Legends ===
    fig.legend(tpr_lines,    class_labels,
               loc="upper center", bbox_to_anchor=(0.375, 0.965),
               title="Class TPRs", ncol=5, frameon=True)
    fig.legend(metric_lines, metric_labels,
               loc="upper center", bbox_to_anchor=(0.815, 0.965),
               title="Global Metrics", ncol=3, frameon=True)

    # save
    out_path = "figures/OFI/real/mnist_ofi_cnn_knn_all_seeds.png"
    make_dirs(out_path)
    plt.savefig(out_path, bbox_inches='tight')

if __name__ == "__main__":
    plot_ofi_mnist()