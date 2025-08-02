from synthetic_datasets import generate_circle_dataset, generate_bar_dataset, \
    generate_ring_dataset, generate_cross_dataset
from OverlapIndex import OverlapIndex
from iCONN_index import iCONN
import os
import numpy as np
from artlib import normalize
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from tqdm import tqdm


def make_dirs(path):
    dir_path = os.path.dirname(path)

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def experiment_circle():
    distance_range = np.linspace(-0.1, 0.5, 40)
    radius = 0.15

    data = analyze_circle_separation_sweep(
        radius=radius,
        distance_range=distance_range,
        n_samples_per_class=50,
        n_seeds=5,
        plot_every_n=5,
        random_state=42
    )
    path = "result_data/synthetic/circles_data.pickle"
    make_dirs(path)
    pickle.dump(data, open(path, "wb"))


def analyze_circle_separation_sweep(
        radius,
        distance_range,
        n_samples_per_class=300,
        n_seeds=10,
        plot_every_n=5,
        random_state=None,
        figsize_per_plot=(3, 3),
):
    print("Analyzing Circle Dataset...")
    rng = np.random.default_rng(random_state)
    distances = []
    sil_scores = []
    db_scores = []
    ch_scores = []
    oi_scores = []
    cn_scores = []

    print("Sweeping clustering metrics...")
    for dist in tqdm(distance_range):
        if dist / 2 + radius > 0.5:
            continue  # Circles would extend beyond unit square

        sil_vals, db_vals, ch_vals, oi_vals, cn_vals = [], [], [], [], []

        for _ in range(n_seeds):
            oi_obj = OverlapIndex(rho=0.9, r_hat=0.1, ART="Fuzzy", match_tracking="MT~")
            conn_obj = iCONN(rho=0.9, match_tracking="MT~")
            # try:
            X, y = generate_circle_dataset(
                radius=radius,
                center_distance=dist,
                n_samples_1=n_samples_per_class,
                n_samples_2=n_samples_per_class,
                random_state=rng.integers(0, 1_000_000)
            )
            X, _, _ = normalize(X)
            sil_vals.append(silhouette_score(X, y))
            db_vals.append(davies_bouldin_score(X, y))
            ch_vals.append(calinski_harabasz_score(X, y))
            oi_vals.append(oi_obj.add_batch(X, y))
            cn_vals.append(conn_obj.add_batch(X, y))

        if sil_vals:
            distances.append(dist)
            sil_scores.append(np.mean(sil_vals))
            db_scores.append(np.mean(db_vals))
            ch_scores.append(np.mean(ch_vals))
            oi_scores.append(np.mean(oi_vals))
            cn_scores.append(np.mean(cn_vals))

    distances = np.array(distances)
    sil_scores = np.array(sil_scores)
    db_scores = np.array(db_scores)
    ch_scores = np.array(ch_scores)
    oi_scores = np.array(oi_scores)
    cn_scores = np.array(cn_scores)

    perfect_sep = 2 * radius

    # === Plot representative circle examples ===
    print("Plotting representative circle examples...")
    selected_dists = distance_range[::plot_every_n]
    num_plots = len(selected_dists)
    fig, axes = plt.subplots(
        1, num_plots,
        figsize=(figsize_per_plot[0] * num_plots, figsize_per_plot[1]),
        sharex=True, sharey=True
    )
    if num_plots == 1:
        axes = [axes]

    for ax, dist in zip(axes, selected_dists):
        if dist / 2 + radius > 0.5:
            ax.set_title(f"dist = {dist:.3f}\n[invalid]")
            ax.axis('off')
            continue

        success = False
        attempts = 0
        while not success and attempts < 10:
            try:
                X, y = generate_circle_dataset(
                    radius=radius,
                    center_distance=dist,
                    n_samples_1=n_samples_per_class,
                    n_samples_2=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                score = silhouette_score(X, y)
                success = True
            except ValueError:
                attempts += 1

        if success:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10, alpha=0.7)
            ax.set_title(f"dist = {dist:.3f}\nscore = {score:.3f}")
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.set_title(f"dist = {dist:.3f}\n[failed]")
            ax.axis('off')

    plt.tight_layout()
    path = "figures/overlap_index/synthetic/blob_separation_exemplars.png"
    make_dirs(path)
    plt.savefig(path)

    data = {"distances": distances, "sil_scores": sil_scores, "db_scores": db_scores,
            "ch_scores": ch_scores, "cn_scores": cn_scores, "oi_scores": oi_scores,
            "perfect_sep": perfect_sep}
    return data

def experiment_ring():
    # Parameters for the inner ring (class 0)
    inner_radius_1 = 0.1
    outer_radius_1 = 0.2

    # Sweep gap values between inner and outer ring (class 1)
    gap_range = np.linspace(-0.2, 0.2, 40)  # e.g., 30 values from 0.01 to 0.3

    # Run the ring separation analysis
    data = analyze_ring_separation_sweep(
        base_inner_radius_1=inner_radius_1,
        base_outer_radius_1=outer_radius_1,
        gap_range=gap_range,
        n_samples_per_class=200,
        n_seeds=10,
        plot_every_n=5,
        random_state=42,
        figsize_per_plot=(3, 3)
    )
    path = "result_data/synthetic/rings_data.pickle"
    make_dirs(path)
    pickle.dump(data, open(path, "wb"))



def analyze_ring_separation_sweep(
        base_inner_radius_1,
        base_outer_radius_1,
        gap_range,
        n_samples_per_class=300,
        n_seeds=10,
        plot_every_n=5,
        random_state=None,
        figsize_per_plot=(3, 3),
):
    print("Analyzing Ring Dataset")
    rng = np.random.default_rng(random_state)
    sil_scores = []
    db_scores = []
    ch_scores = []
    oi_scores = []
    cn_scores = []
    gaps = []
    radius_gaps = []

    print("Sweeping clustering metrics...")
    for gap in tqdm(gap_range):
        sil_vals, db_vals, ch_vals, oi_vals, cn_vals = [], [], [], [], []
        inner_radius_2 = base_outer_radius_1 + gap
        outer_radius_2 = inner_radius_2 + (base_outer_radius_1 - base_inner_radius_1)
        radius_gap = 0.5 * (inner_radius_2 + outer_radius_2) - 0.5 * (
                    base_inner_radius_1 + base_outer_radius_1)
        for _ in range(n_seeds):
            oi_obj = OverlapIndex(rho=0.9, r_hat=0.1, ART="Fuzzy", match_tracking="MT~")
            conn_obj = iCONN(rho=0.9, match_tracking="MT~")
            try:
                X, y = generate_ring_dataset(
                    inner_radius_1=base_inner_radius_1,
                    outer_radius_1=base_outer_radius_1,
                    inner_radius_2=inner_radius_2,
                    outer_radius_2=outer_radius_2,
                    n_samples_1=n_samples_per_class,
                    n_samples_2=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                X, _, _ = normalize(X)
                sil_vals.append(silhouette_score(X, y))
                db_vals.append(davies_bouldin_score(X, y))
                ch_vals.append(calinski_harabasz_score(X, y))
                oi_vals.append(oi_obj.add_batch(X, y))
                cn_vals.append(conn_obj.add_batch(X, y))
            except ValueError:
                continue

        if sil_vals:
            gaps.append(gap)
            radius_gaps.append(radius_gap)
            sil_scores.append(np.mean(sil_vals))
            db_scores.append(np.mean(db_vals))
            ch_scores.append(np.mean(ch_vals))
            oi_scores.append(np.mean(oi_vals))
            cn_scores.append(np.mean(cn_vals))

    # Convert lists to arrays for plotting
    gaps = np.array(gaps)
    sil_scores = np.array(sil_scores)
    db_scores = np.array(db_scores)
    ch_scores = np.array(ch_scores)
    oi_scores = np.array(oi_scores)
    cn_scores = np.array(cn_scores)

    perfect_sep = 0  # Perfect separation happens at zero gap


    # === Plot representative examples ===
    print("Plotting representative ring examples...")
    selected_seps = gap_range[::plot_every_n]
    num_plots = len(selected_seps)
    fig, axes = plt.subplots(
        1, num_plots,
        figsize=(figsize_per_plot[0] * num_plots, figsize_per_plot[1]),
        sharex=True, sharey=True
    )
    if num_plots == 1:
        axes = [axes]

    for ax, gap in zip(axes, selected_seps):
        success = False
        attempts = 0
        while not success and attempts < 10:
            try:
                inner_radius_2 = base_outer_radius_1 + gap
                outer_radius_2 = inner_radius_2 + (
                            base_outer_radius_1 - base_inner_radius_1)
                X, y = generate_ring_dataset(
                    inner_radius_1=base_inner_radius_1,
                    outer_radius_1=base_outer_radius_1,
                    inner_radius_2=inner_radius_2,
                    outer_radius_2=outer_radius_2,
                    n_samples_1=n_samples_per_class,
                    n_samples_2=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                score = silhouette_score(X, y)
                success = True
            except ValueError:
                attempts += 1

        if success:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10, alpha=0.7)
            ax.set_title(f"sep = {gap:.3f}\nscore = {score:.3f}")
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.set_title(f"sep = {gap:.3f}\n[failed]")
            ax.axis('off')

    plt.tight_layout()
    path = "figures/overlap_index/synthetic/ring_separation_exemplars.png"
    make_dirs(path)
    plt.savefig(path)
    data = {"distances": np.array(radius_gaps), "sil_scores": sil_scores,
            "db_scores": db_scores, "ch_scores": ch_scores, "cn_scores": cn_scores,
            "oi_scores": oi_scores,
            "perfect_sep": (base_outer_radius_1 - base_inner_radius_1)}
    return data

def experiment_bars():
    # Parameters for the bar
    bar_width = 0.2
    bar_height = 0.4
    separation_range = np.linspace(-0.1, 0.5,
                                   40)  # from almost overlap to strong separation

    # Run analysis
    data = analyze_bar_separation_sweep(
        bar_width=bar_width,
        bar_height=bar_height,
        separation_range=separation_range,
        orientation="vertical",  # or "horizontal"
        n_samples_per_class=50,
        n_seeds=10,
        plot_every_n=5,
        random_state=42,
        figsize_per_plot=(3, 3)
    )
    path = "result_data/synthetic/bars_data.pickle"
    make_dirs(path)
    pickle.dump(data, open(path, "wb"))



def analyze_bar_separation_sweep(
    bar_width,
    bar_height,
    separation_range,
    orientation="vertical",
    n_samples_per_class=300,
    n_seeds=10,
    plot_every_n=5,
    random_state=None,
    figsize_per_plot=(3, 3),
):
    print("Analyzing Bar Dataset")
    rng = np.random.default_rng(random_state)
    separations = []
    sil_scores = []
    db_scores = []
    ch_scores = []
    oi_scores = []
    cn_scores = []

    print("Sweeping clustering metrics...")
    for sep in tqdm(separation_range):
        sil_vals, db_vals, ch_vals, oi_vals, cn_vals = [], [], [], [], []

        for _ in range(n_seeds):
            oi_obj = OverlapIndex(rho=0.9, r_hat=0.1, ART="Fuzzy", match_tracking="MT+")
            conn_obj = iCONN(rho=0.9, match_tracking="MT~")
            try:
                X, y = generate_bar_dataset(
                    bar_width=bar_width,
                    bar_height=bar_height,
                    separation=sep,
                    orientation=orientation,
                    n_samples_1=n_samples_per_class,
                    n_samples_2=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                X, _, _ = normalize(X)
                sil_vals.append(silhouette_score(X, y))
                db_vals.append(davies_bouldin_score(X, y))
                ch_vals.append(calinski_harabasz_score(X, y))
                oi_vals.append(oi_obj.add_batch(X, y))
                cn_vals.append(conn_obj.add_batch(X,y))
            except ValueError:
                continue

        if sil_vals:
            separations.append(sep)
            sil_scores.append(np.mean(sil_vals))
            db_scores.append(np.mean(db_vals))
            ch_scores.append(np.mean(ch_vals))
            oi_scores.append(np.mean(oi_vals))
            cn_scores.append(np.mean(cn_vals))

    # === Convert lists to arrays ===
    separations = np.array(separations)
    sil_scores = np.array(sil_scores)
    db_scores = np.array(db_scores)
    ch_scores = np.array(ch_scores)
    oi_scores = np.array(oi_scores)
    cn_scores = np.array(cn_scores)

    # === Define perfect separation based on orientation ===
    perfect_sep = bar_width if orientation == "vertical" else bar_height


    # === Plot representative examples ===
    print("Plotting representative bar examples...")
    selected_seps = separation_range[::plot_every_n]
    num_plots = len(selected_seps)
    fig, axes = plt.subplots(
        1, num_plots,
        figsize=(figsize_per_plot[0] * num_plots, figsize_per_plot[1]),
        sharex=True, sharey=True
    )
    if num_plots == 1:
        axes = [axes]

    for ax, sep in zip(axes, selected_seps):
        success = False
        attempts = 0
        while not success and attempts < 10:
            try:
                X, y = generate_bar_dataset(
                    bar_width=bar_width,
                    bar_height=bar_height,
                    separation=sep,
                    orientation=orientation,
                    n_samples_1=n_samples_per_class,
                    n_samples_2=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                score = silhouette_score(X, y)
                success = True
            except ValueError:
                attempts += 1

        if success:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10, alpha=0.7)
            ax.set_title(f"sep = {sep:.3f}\nscore = {score:.3f}")
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.set_title(f"sep = {sep:.3f}\n[failed]")
            ax.axis('off')

    plt.tight_layout()
    path = "figures/overlap_index/synthetic/bar_separation_exemplars.png"
    make_dirs(path)
    plt.savefig(path)
    data = {"distances": separations, "sil_scores":sil_scores, "db_scores":db_scores, "ch_scores":ch_scores, "cn_scores":cn_scores, "oi_scores":oi_scores, "perfect_sep": perfect_sep}
    return data


def experiment_cross():
    # Parameters for the cross
    bar_width = 0.1
    bar_height = 0.3
    offset_range = np.linspace(-0.3, 0.3,
                               40)  # from overlap (bottom-heavy T) to top-heavy T

    # Run analysis
    data = analyze_cross_offset_sweep(
        bar_width=bar_width,
        bar_height=bar_height,
        offset_range=offset_range,
        n_samples_per_class=100,
        n_seeds=10,
        plot_every_n=5,
        random_state=42,
        figsize_per_plot=(3, 3)
    )
    path = "result_data/synthetic/cross_data.pickle"
    make_dirs(path)
    pickle.dump(data, open(path, "wb"))


def analyze_cross_offset_sweep(
    bar_width,
    bar_height,
    offset_range,
    n_samples_per_class=300,
    n_seeds=10,
    plot_every_n=5,
    random_state=None,
    figsize_per_plot=(3, 3),
):
    print("Analyzing Cross Dataset...")
    rng = np.random.default_rng(random_state)
    offsets = []
    sil_scores = []
    db_scores = []
    ch_scores = []
    oi_scores = []
    cn_scores = []

    print("Sweeping clustering metrics...")
    for offset in tqdm(offset_range):
        sil_vals, db_vals, ch_vals, oi_vals, cn_vals = [], [], [], [], []

        for _ in range(n_seeds):
            oi_obj = OverlapIndex(rho=0.9, r_hat=0.1, ART="Fuzzy", match_tracking="MT~")
            conn_obj = iCONN(rho=0.9, match_tracking="MT~")
            try:
                X, y = generate_cross_dataset(
                    bar_width=bar_width,
                    bar_height=bar_height,
                    offset=offset,
                    n_samples_vert=n_samples_per_class,
                    n_samples_horiz=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                X, _, _ = normalize(X)
                sil_vals.append(silhouette_score(X, y))
                db_vals.append(davies_bouldin_score(X, y))
                ch_vals.append(calinski_harabasz_score(X, y))
                oi_vals.append(oi_obj.add_batch(X, y))
                cn_vals.append(conn_obj.add_batch(X, y))
            except ValueError:
                continue

        if sil_vals:
            offsets.append(offset)
            sil_scores.append(np.mean(sil_vals))
            db_scores.append(np.mean(db_vals))
            ch_scores.append(np.mean(ch_vals))
            oi_scores.append(np.mean(oi_vals))
            cn_scores.append(np.mean(cn_vals))

    # === Convert lists to arrays ===
    offsets = np.array(offsets)
    sil_scores = np.array(sil_scores)
    db_scores = np.array(db_scores)
    ch_scores = np.array(ch_scores)
    oi_scores = np.array(oi_scores)
    cn_scores = np.array(cn_scores)

    # === Define reference point: 0 offset is perfect "+"
    perfect_offset = 0.5 * bar_width + 0.5 * bar_height


    # === Plot representative examples ===
    print("Plotting representative cross examples...")
    selected_offsets = offset_range[::plot_every_n]
    num_plots = len(selected_offsets)
    fig, axes = plt.subplots(
        1, num_plots,
        figsize=(figsize_per_plot[0] * num_plots, figsize_per_plot[1]),
        sharex=True, sharey=True
    )
    if num_plots == 1:
        axes = [axes]

    for ax, offset in zip(axes, selected_offsets):
        success = False
        attempts = 0
        while not success and attempts < 10:
            try:
                X, y = generate_cross_dataset(
                    bar_width=bar_width,
                    bar_height=bar_height,
                    offset=offset,
                    n_samples_vert=n_samples_per_class,
                    n_samples_horiz=n_samples_per_class,
                    random_state=rng.integers(0, 1_000_000)
                )
                score = silhouette_score(X, y)
                success = True
            except ValueError:
                attempts += 1

        if success:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10, alpha=0.7)
            ax.set_title(f"offset = {offset:.3f}\nscore = {score:.3f}")
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.set_title(f"offset = {offset:.3f}\n[failed]")
            ax.axis('off')

    plt.tight_layout()
    path = "figures/overlap_index/synthetic/cross_offset_exemplars.png"
    make_dirs(path)
    plt.savefig(path)

    data = {
        "offsets": offsets,
        "sil_scores": sil_scores,
        "db_scores": db_scores,
        "ch_scores": ch_scores,
        "cn_scores": cn_scores,
        "oi_scores": oi_scores,
        "perfect_offset": perfect_offset
    }
    return data


if __name__ == "__main__":
    experiment_circle()
    experiment_ring()
    experiment_bars()
    experiment_cross()
