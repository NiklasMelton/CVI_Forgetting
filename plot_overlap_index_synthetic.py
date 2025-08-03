import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import ScalarFormatter, FixedLocator
from synthetic_datasets import generate_circle_dataset, generate_bar_dataset, \
    generate_ring_dataset, generate_cross_dataset
from common import make_dirs

def plot_oi_synthetic_combined_indices():
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    # Color-blind–friendly palette (Okabe-Ito)
    colorblind_palette = [
        "#E69F00",  # orange
        # "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        # "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]

    # Filenames and subplot titles
    files = [
        ("result_data/overlap_index/synthetic/circles_data.pickle", "Circles"),
        ("result_data/overlap_index/synthetic/rings_data.pickle", "Rings"),
        ("result_data/overlap_index/synthetic/bars_data.pickle", "Bars"),
        ("result_data/overlap_index/synthetic/cross_data.pickle", "Cross"),
    ]

    # Score labels and keys
    score_keys = [
        ("sil_scores", "Silhouette"),
        ("db_scores", "Davies-Bouldin"),
        ("ch_scores", "Calinski–Harabasz"),
        ("cn_scores", "CONN"),
        ("oi_scores", "Overlap Index"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for ax, (filename, title) in zip(axs, files):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        if title == "Cross":
            distances = data["offsets"]
            perfect_sep = data["perfect_offset"]
        else:
            distances = data["distances"]
            perfect_sep = data["perfect_sep"]

        # Plot each score line with color-blind–friendly colors
        for i, (key, label) in enumerate(score_keys):
            color = colorblind_palette[i % len(colorblind_palette)]
            if label != "Overlap Index":
                ax.plot(distances, data[key], label=label, alpha=0.7, color=color)
            else:
                ax.plot(distances, data[key], label=label, color=color, linewidth=3)

        # Plot perfect separation line(s)
        if title in ["Cross", "Rings"]:
            ax.axvline(perfect_sep, color='gray', linestyle='--', label="Perfect Sep +")
            ax.axvline(-perfect_sep, color='gray', linestyle='--', label="Perfect Sep –")
        else:
            ax.axvline(perfect_sep, color='gray', linestyle='--', label="Perfect Separation")

        ax.set_title(title)
        ax.set_xlabel("Radius Difference" if title == "Rings" else "Center Distance")
        ax.set_ylabel("Score")

        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, None)
        ymax = ax.get_ylim()[1]

        # Minor ticks: linear from 0–1 and log from 1 onward (including 1–10)
        linear_minors = [i / 10 for i in range(1, 10)]  # 0.1 to 0.9
        log_minors = []

        # Add log minors from 1–10
        log_minors += [i for i in range(2, 10) if i < ymax]

        # Add log minors from 10 upwards
        for decade in range(1, int(np.log10(ymax)) + 2):
            base = 10 ** decade
            log_minors += [base * f for f in range(2, 10) if base * f < ymax]

        all_minor_ticks = linear_minors + log_minors
        ax.yaxis.set_minor_locator(FixedLocator(all_minor_ticks))

        # Ticks visible, grid only for major
        ax.tick_params(axis='y', which='minor', length=4, width=0.8)
        ax.tick_params(axis='y', which='major', length=7, width=1.2)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.grid(False, which='minor')

        # Format major ticks as plain numbers
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        frameon=True,
        title="Scores"
    )

    plt.tight_layout()
    path = "figures/overlap_index/synthetic/combined_datasets_indices.png"
    make_dirs(path)
    plt.savefig(path, bbox_inches='tight')


def plot_combined_exemplars():

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    # Assumed constants – update as needed
    n_samples_per_class = 300
    rng = np.random.default_rng(42)

    # Parameters for each dataset
    radius = 0.15
    circle_dist = 0.4

    base_inner_radius_1 = 0.1
    base_outer_radius_1 = 0.2
    inner_radius_2 = 0.25
    outer_radius_2 = 0.35

    bar_width = 0.2
    bar_height = 0.4
    bar_sep = 0.25
    orientation = "vertical"

    cross_bar_width = 0.1
    cross_bar_height = 0.3
    cross_offset = 0.25

    # === Generate datasets ===
    X_circle, y_circle = generate_circle_dataset(
        radius=radius,
        center_distance=circle_dist,
        n_samples_1=50,
        n_samples_2=50,
        random_state=42
    )

    X_ring, y_ring = generate_ring_dataset(
        inner_radius_1=base_inner_radius_1,
        outer_radius_1=base_outer_radius_1,
        inner_radius_2=inner_radius_2,
        outer_radius_2=outer_radius_2,
        n_samples_1=200,
        n_samples_2=200,
        random_state=42
    )

    X_bar, y_bar = generate_bar_dataset(
        bar_width=bar_width,
        bar_height=bar_height,
        separation=bar_sep,
        orientation=orientation,
        n_samples_1=50,
        n_samples_2=50,
        random_state=42
    )

    X_cross, y_cross = generate_cross_dataset(
        bar_width=cross_bar_width,
        bar_height=cross_bar_height,
        offset=cross_offset,
        n_samples_vert=100,
        n_samples_horiz=100,
        random_state=42
    )

    # === Plotting ===
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axs = axs.flatten()

    datasets = [
        (X_circle, y_circle, "Circles"),
        (X_ring, y_ring, "Rings"),
        (X_bar, y_bar, "Bars"),
        (X_cross, y_cross, "Cross"),
    ]

    xlim = (0.1, 0.9)
    ylim = (0.1, 0.9)

    for ax, (X, y, title) in zip(axs, datasets):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

        if title == "Rings":
            r1 = 0.5 * (base_inner_radius_1 + base_outer_radius_1)
            r2 = 0.5 * (inner_radius_2 + outer_radius_2)
            y_pos = 0.5
            start = (0.5 + r1, y_pos)
            end = (0.5 + r2, y_pos)
            ax.annotate(
                "", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="<->", color='black', linewidth=4)
            )
            dist = abs(end[0] - start[0])

        elif title == "Circles":
            center1 = np.array([0.5 - 0.5 * circle_dist, 0.5])
            center2 = np.array([0.5 + 0.5 * circle_dist, 0.5])
            ax.annotate(
                "", xy=center2, xytext=center1,
                arrowprops=dict(arrowstyle="<->", color='black', linewidth=4)
            )
            dist = np.linalg.norm(center2 - center1)

        elif title == "Bars":
            if orientation == "vertical":
                x1 = 0.5 - 0.5 * bar_sep
                x2 = 0.5 + 0.5 * bar_sep
                y_val = 0.5
                ax.annotate(
                    "", xy=(x2, y_val), xytext=(x1, y_val),
                    arrowprops=dict(arrowstyle="<->", color='black', linewidth=4)
                )
                dist = abs(x2 - x1)
            else:
                y1 = 0.5 - 0.5 * bar_sep
                y2 = 0.5 + 0.5 * bar_sep
                x_val = 0.5
                ax.annotate(
                    "", xy=(x_val, y2), xytext=(x_val, y1),
                    arrowprops=dict(arrowstyle="<->", color='black', linewidth=4)
                )
                dist = abs(y2 - y1)

        elif title == "Cross":
            y1 = 0.5
            y2 = 0.5 + cross_offset
            x_val = 0.5
            ax.annotate(
                "", xy=(x_val, y2), xytext=(x_val, y1),
                arrowprops=dict(arrowstyle="<->", color='black', linewidth=4)
            )
            dist = abs(y2 - y1)

        # Annotate distance above the plot
        if title == "Rings":
            ax.text(0.5, 0.97, f"Radius Difference = {dist:.3f}", ha='center', va='top',
                    transform=ax.transAxes, fontsize=14)
        else:
            ax.text(0.5, 0.97, f"Center Distance = {dist:.3f}", ha='center', va='top',
                    transform=ax.transAxes, fontsize=14)
    path = "figures/overlap_index/synthetic/combined_exemplars.png"
    make_dirs(path)
    plt.savefig(path, bbox_inches='tight')


if __name__ == "__main__":
    plot_oi_synthetic_combined_indices()
    plot_combined_exemplars()