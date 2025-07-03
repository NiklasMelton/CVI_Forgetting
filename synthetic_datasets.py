import numpy as np

def generate_ring_dataset(
    inner_radius_1, outer_radius_1,
    inner_radius_2, outer_radius_2,
    n_samples_1, n_samples_2,
    center=(0.5, 0.5), random_state=None
):
    """
    Generate a 2D dataset with two classes: ring within a ring.

    Parameters:
    - inner_radius_1, outer_radius_1: float
        Inner and outer radius for class 1 (label 0).
    - inner_radius_2, outer_radius_2: float
        Inner and outer radius for class 2 (label 1).
    - n_samples_1, n_samples_2: int
        Number of samples to generate for each class.
    - center: tuple of float
        Center of the rings (default is (0.5, 0.5)).
    - random_state: int or None
        Seed for reproducibility.

    Returns:
    - X: np.ndarray, shape (n_samples_1 + n_samples_2, 2)
        2D coordinates of the samples.
    - y: np.ndarray, shape (n_samples_1 + n_samples_2,)
        Labels (0 or 1).
    """
    rng = np.random.default_rng(random_state)

    def generate_ring(inner_r, outer_r, n):
        if not (0 <= inner_r < outer_r <= 1.0):
            raise ValueError(f"Invalid ring parameters: 0 < {inner_r=} < {outer_r=} <= 1.0 required.")
        theta = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(rng.uniform(inner_r**2, outer_r**2, n))  # Uniform area density
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        return np.stack((x, y), axis=1)

    X1 = generate_ring(inner_radius_1, outer_radius_1, n_samples_1)
    X2 = generate_ring(inner_radius_2, outer_radius_2, n_samples_2)

    y1 = np.zeros(n_samples_1, dtype=int)
    y2 = np.ones(n_samples_2, dtype=int)

    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))

    return X, y


def generate_circle_dataset(
    radius,
    center_distance,
    n_samples_1,
    n_samples_2,
    random_state=None
):
    """
    Generate a 2-class dataset of uniform circular blobs.

    Parameters:
    - radius: float
        Radius of both circular classes (must be > 0).
    - center_distance: float
        Horizontal distance between the two circle centers.
    - n_samples_1, n_samples_2: int
        Number of samples in each class.
    - random_state: int or None
        For reproducibility.

    Returns:
    - X: np.ndarray of shape (n_samples_1 + n_samples_2, 2)
        The 2D points.
    - y: np.ndarray of shape (n_samples_1 + n_samples_2,)
        Class labels (0 or 1).
    """
    if radius <= 0:
        raise ValueError("Radius must be greater than 0.")

    # Compute circle centers based on horizontal offset (centered around 0.5)
    cx1 = 0.5 - center_distance / 2
    cx2 = 0.5 + center_distance / 2
    cy = 0.5

    # Ensure both circles stay within [0.0, 1.0]^2
    if not (radius <= cx1 <= 1 - radius and radius <= cx2 <= 1 - radius):
        raise ValueError("Circles extend beyond horizontal bounds.")
    if not (radius <= cy <= 1 - radius):
        raise ValueError("Circles extend beyond vertical bounds.")

    rng = np.random.default_rng(random_state)

    def sample_circle(n, center_x, center_y):
        theta = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(rng.uniform(0, radius ** 2, n))
        x = center_x + r * np.cos(theta)
        y = center_y + r * np.sin(theta)
        return np.stack((x, y), axis=1)

    X1 = sample_circle(n_samples_1, cx1, cy)
    X2 = sample_circle(n_samples_2, cx2, cy)
    y1 = np.zeros(n_samples_1, dtype=int)
    y2 = np.ones(n_samples_2, dtype=int)

    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))

    return X, y


def generate_bar_dataset(
        bar_width,
        bar_height,
        separation,
        orientation="vertical",  # "vertical" or "horizontal"
        n_samples_1=500,
        n_samples_2=500,
        random_state=None
):
    """
    Generate two rectangular bars uniformly filled, separated by a given distance.

    Parameters:
    - bar_width: float
        Width of each bar (x-axis if vertical, y-axis if horizontal).
    - bar_height: float
        Height of each bar (y-axis if vertical, x-axis if horizontal).
    - separation: float
        Distance between the centers of the two bars (along separation axis).
    - orientation: str
        "vertical" (bars left/right of center) or "horizontal" (bars above/below center).
    - n_samples_1, n_samples_2: int
        Number of points in each bar.
    - random_state: int or None
        Seed for reproducibility.

    Returns:
    - X: np.ndarray of shape (n_samples_1 + n_samples_2, 2)
    - y: np.ndarray of shape (n_samples_1 + n_samples_2,)
    """
    assert orientation in {"vertical",
                           "horizontal"}, "Orientation must be 'vertical' or 'horizontal'"
    assert bar_width > 0 and bar_height > 0, "Bar dimensions must be positive"

    rng = np.random.default_rng(random_state)
    cx, cy = 0.5, 0.5

    if orientation == "vertical":
        # Horizontal separation
        cx1 = cx - separation / 2
        cx2 = cx + separation / 2
        cy1 = cy2 = cy

        # Check bounds
        if not (0 <= cx1 - bar_width / 2 and cx2 + bar_width / 2 <= 1.0):
            raise ValueError("Bars extend beyond horizontal bounds.")
        if not (0 <= cy - bar_height / 2 and cy + bar_height / 2 <= 1.0):
            raise ValueError("Bars extend beyond vertical bounds.")

        def sample_bar(n, center_x):
            x = rng.uniform(center_x - bar_width / 2, center_x + bar_width / 2, n)
            y = rng.uniform(cy - bar_height / 2, cy + bar_height / 2, n)
            return np.stack((x, y), axis=1)

    else:  # horizontal
        # Vertical separation
        cy1 = cy - separation / 2
        cy2 = cy + separation / 2
        cx1 = cx2 = cx

        # Check bounds
        if not (0 <= cy1 - bar_width / 2 and cy2 + bar_width / 2 <= 1.0):
            raise ValueError("Bars extend beyond vertical bounds.")
        if not (0 <= cx - bar_height / 2 and cx + bar_height / 2 <= 1.0):
            raise ValueError("Bars extend beyond horizontal bounds.")

        def sample_bar(n, center_y):
            x = rng.uniform(cx - bar_height / 2, cx + bar_height / 2, n)
            y = rng.uniform(center_y - bar_width / 2, center_y + bar_width / 2, n)
            return np.stack((x, y), axis=1)

    # Generate both bars
    X1 = sample_bar(n_samples_1, cx1 if orientation == "vertical" else cy1)
    X2 = sample_bar(n_samples_2, cx2 if orientation == "vertical" else cy2)

    y1 = np.zeros(n_samples_1, dtype=int)
    y2 = np.ones(n_samples_2, dtype=int)

    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    return X, y
