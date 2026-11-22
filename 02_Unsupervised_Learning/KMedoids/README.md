# K-Medoids Clustering

## Files in this folder

| File | Description |
|---|---|
| `8_k_medoids.ipynb` | K-Medoids with 2 clusters and Manhattan distance on 10 inline points; asks for a new point, predicts its cluster and plots it |

## 1. Algorithm Name

K-Medoids Clustering (PAM-style partitioning), unsupervised learning, clustering.

## 2. Definition

K-Medoids divides data into `K` clusters like K-Means, but each cluster centre is an actual data point (the medoid) instead of an average. Because it only needs pairwise distances, it can use any distance metric, such as Manhattan.

## 3. How It Works

1. Choose `K` initial medoids from the data points.
2. Assign every point to its nearest medoid under the chosen metric.
3. Within each cluster, choose as the new medoid the point with the smallest total distance to the other members. The `'pam'` method instead tries swapping medoids with non-medoids.
4. Repeat until the medoids stop changing or `max_iter` is reached.

## 4. Why It Is Used

- It is more robust to outliers than K-Means, because a medoid cannot be pulled away from the data by extreme values.
- It works with any distance or dissimilarity measure, including non-Euclidean ones.
- Each cluster centre is a real, interpretable example from the data.

## 5. Mathematical Intuition

- Objective: minimise `sum over points d(x_i, m_c(i))`, the total distance from each point to its cluster's medoid. K-Means minimises squared Euclidean distance to a mean instead.
- Manhattan distance (used here): `d(a, b) = |a_x - b_x| + |a_y - b_y|`. It sums absolute differences, so it is less dominated by one large difference than squared Euclidean distance.
- Because only sums of distances are compared, the medoid behaves like a multi-dimensional median.

## 6. Important Parameters

**General (`sklearn_extra.cluster.KMedoids`)**

| Parameter | Default | Meaning |
|---|---|---|
| `n_clusters` | `8` | Number of clusters `K` |
| `metric` | `'euclidean'` | Distance metric (`'manhattan'`, `'cosine'`, `'precomputed'`, ...) |
| `method` | `'alternate'` | `'alternate'` (fast, K-Means-like) or `'pam'` (slower, often better) |
| `init` | `'heuristic'` | `'heuristic'`, `'k-medoids++'`, `'random'`, `'build'` |
| `max_iter` | `300` | Maximum iterations |
| `random_state` | `None` | Seed for random initialisation |

**Used in this notebook**

- `KMedoids(n_clusters=2, metric='manhattan')`; all other parameters at their defaults.
- The printed and plotted cluster numbers are `label + 1`, so they read "Cluster 1" and "Cluster 2" instead of 0 and 1.

## 7. Advantages

- Robust to outliers and noise.
- Works with any distance metric.
- Cluster centres are real data points.

## 8. Disadvantages

- Slower than K-Means. PAM needs pairwise distances, roughly `O(n^2)` memory and time.
- `K` must still be chosen in advance.
- Like K-Means, it favours compact clusters and can find local optima.
- Requires the extra `scikit-learn-extra` package.

## 9. Real-World Applications

- Clustering with custom distances, such as edit distance between strings or travel distance on a road grid.
- Choosing representative examples (prototypes) from a dataset.
- Customer or patient grouping when outliers are common.
- Facility location problems, where centres must be existing sites.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Building the DataFrame |
| `numpy` | Building the new point as an array |
| `sklearn_extra.cluster.KMedoids` | The model (package `scikit-learn-extra`) |
| `matplotlib.pyplot` | Scatter plot |

## 11. Dataset Information

- Inline data defined in the notebook: 10 rows x 2 columns.
- `x = [2, 3, 3, 4, 6, 6, 7, 7, 8, 7]`, `y = [6, 4, 8, 7, 2, 4, 3, 4, 5, 6]`.
- No external file is needed.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Install the optional package, as noted in `requirements.txt`. It ships wheels only for older Python versions, so use Python 3.10 for this notebook:
   ```
   python -m pip install "numpy<2" "scikit-learn-extra>=0.3.0"
   ```
3. Run `python -m jupyter notebook` from the repository root and open `02_Unsupervised_Learning/KMedoids/8_k_medoids.ipynb`.
4. Run the cells from top to bottom. The input cell asks:

| Prompt | What to type |
|---|---|
| `Enter X coordinate of the new point:` | a number, e.g. `4` (data range 2 to 8) |
| `Enter Y coordinate of the new point:` | a number, e.g. `6` (data range 2 to 8) |

A non-numeric entry raises a `ValueError`.

If the import cell shows `ValueError: numpy.dtype size changed, may indicate binary incompatibility`, the installed `scikit-learn-extra` binary was built against a different NumPy version from the one in the environment. A separate environment with a NumPy version compatible with that `scikit-learn-extra` build is the usual way around it.

## 13. Expected Output

- After the input: `The point (x, y) belongs to Cluster <1 or 2>`.
- A scatter plot with the two clusters in red and blue, the two medoids as black `X` markers, and the new point as a yellow star labelled with its cluster.

Notes on the saved outputs:

- The first cell (imports) has a saved `ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`. The clustering cell has a saved `NameError: name 'KMedoids' is not defined`, which follows from the failed import.
- The later cells still have saved outputs: `The point (4.0, 6.0) belongs to Cluster 1` and the scatter plot. Those outputs therefore appear to come from a different kernel session from the one that recorded the errors.
