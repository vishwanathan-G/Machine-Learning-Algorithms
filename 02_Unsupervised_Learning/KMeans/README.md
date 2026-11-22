# K-Means Clustering

## Files in this folder

| File | Description |
|---|---|
| `7_k_means.ipynb` | K-Means with 4 clusters on 8 inline points; asks for a new point, predicts its cluster and plots it |
| `kmeans_50000_clustering.ipynb` | K-Means on 50,000 points: exploration, elbow method and silhouette score for K = 2 to 8, final model with K = 5, evaluation, new-point prediction, and saving the results. The notebook has no saved outputs. |
| `kmeans_50000_data.csv` | 50,000 rows x 3 columns (`X`, `Y`, `Original_Group`) |

## 1. Algorithm Name

K-Means Clustering, unsupervised learning, clustering.

## 2. Definition

K-Means divides data into `K` groups (clusters) so that each point belongs to the cluster with the nearest centre (centroid). It needs no labels. The groups are found from the data alone.

## 3. How It Works

1. Choose `K` and place `K` initial centroids. scikit-learn uses `k-means++`, which spreads the starting centroids apart.
2. **Assign** each point to its nearest centroid (Euclidean distance).
3. **Update** each centroid to the mean of the points assigned to it.
4. Repeat steps 2 and 3 until the assignments stop changing or `max_iter` is reached.
5. Because the result depends on the starting centroids, the algorithm is run `n_init` times and the run with the lowest inertia is kept.

## 4. Why It Is Used

- It is fast and scales to large datasets.
- The result is easy to explain: every cluster is summarised by its centroid.
- Common uses include segmentation, compression and pre-processing for other models.

## 5. Mathematical Intuition

- Objective (inertia, or within-cluster sum of squares): `J = sum over points ||x_i - mu_c(i)||^2`, where `mu_c(i)` is the centroid of the point's cluster. Each assign and update step can only lower `J`, so the algorithm converges, though possibly to a local minimum.
- Elbow method: plot inertia against `K`. Inertia always decreases as `K` grows. The "elbow", where the decrease slows sharply, suggests a reasonable `K`.
- Silhouette score of a point: `s = (b - a) / max(a, b)`, where `a` is the mean distance to points in its own cluster and `b` is the mean distance to points in the nearest other cluster. It ranges from -1 to 1, and higher is better.

## 6. Important Parameters

**General (`sklearn.cluster.KMeans`)**

| Parameter | Default | Meaning |
|---|---|---|
| `n_clusters` | `8` | Number of clusters `K` |
| `init` | `'k-means++'` | How initial centroids are chosen |
| `n_init` | `'auto'` | Number of runs with different starting centroids |
| `max_iter` | `300` | Maximum assign/update iterations per run |
| `tol` | `1e-4` | Convergence tolerance |
| `random_state` | `None` | Seed for reproducible results |

**Used in these notebooks**

- `7_k_means.ipynb`: `KMeans(n_clusters=4)` with no other parameters. There is no `random_state`, so cluster numbering can change between runs. The plot loop runs `for i in range(kmeans.n_clusters)` with the colours `['red', 'green', 'blue', 'hotpink']`, one per cluster.
- `kmeans_50000_clustering.ipynb`:
  - Model selection: `KMeans(n_clusters=k, random_state=42, n_init=10)` for `k` from 2 to 8. The silhouette score is computed on a reproducible sample of 5,000 points (`random_state=42`).
  - Final model: `KMeans(n_clusters=5, random_state=42, n_init=10)` (`N_CLUSTERS = 5`).
  - Only `X` and `Y` are used. `Original_Group` is not given to the model.

## 7. Advantages

- Simple, fast and memory-efficient; works on very large datasets.
- Easy to interpret through centroids.
- Assigns new points easily with `predict`.
- Converges quickly in practice.

## 8. Disadvantages

- `K` must be chosen in advance.
- Assumes roughly round, similarly sized clusters; struggles with elongated or irregular shapes (see `../DBSCAN`).
- Sensitive to outliers, because means are pulled by extreme values.
- Sensitive to feature scale and to the initial centroids.
- Cluster numbers are arbitrary labels.

## 9. Real-World Applications

- Customer segmentation for marketing.
- Image colour quantisation and compression.
- Grouping documents or products by similarity.
- Choosing warehouse or store locations from delivery points.

## 10. Python Libraries Used

| Library | Used for | Notebooks |
|---|---|---|
| `pandas` | DataFrames, CSV loading and saving | both |
| `numpy` | Colour map, `argmax` of silhouette scores | `kmeans_50000_clustering` |
| `matplotlib.pyplot` | Scatter plots, elbow and silhouette plots | both |
| `sklearn.cluster.KMeans` | The model | both |
| `sklearn.metrics.silhouette_score` | Cluster quality | `kmeans_50000_clustering` |
| `sklearn.preprocessing.StandardScaler` | Imported, not used | `kmeans_50000_clustering` |
| `joblib` | Saving the fitted model | `kmeans_50000_clustering` |

## 11. Dataset Information

| Notebook | Data | Size | Notes |
|---|---|---|---|
| `7_k_means.ipynb` | Inline Python dictionary | 8 rows x 2 columns | `X = [2, 2, 8, 5, 7, 6, 1, 4]`, `Y = [10, 5, 4, 8, 5, 4, 2, 9]` |
| `kmeans_50000_clustering.ipynb` | `kmeans_50000_data.csv` | 50,000 rows x 3 columns | `X` (about -1.6 to 11.8), `Y` (about -1.3 to 11.7), `Original_Group` (`Group_1` to `Group_5`, 10,000 rows each). No missing values. The notebook describes it as generated (synthetic) practice data. |

`kmeans_50000_clustering.ipynb` writes two files when run: `kmeans_50000_with_clusters.csv` and `kmeans_model.joblib`. Both are listed in `.gitignore`, so they are not committed.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started). `joblib` is included in `requirements.txt`.
2. Run `python -m jupyter notebook` from the repository root and open a notebook in `02_Unsupervised_Learning/KMeans/`.
3. Run the cells from top to bottom. Both notebooks ask for a new point:

| Prompt | What to type |
|---|---|
| `Enter X coordinate of the new point:` | a number, e.g. `6` (inline data: 1 to 8; 50,000-point data: about -1.6 to 11.8) |
| `Enter Y coordinate of the new point:` | a number, e.g. `4` (inline data: 2 to 10; 50,000-point data: about -1.3 to 11.7) |

A non-numeric entry raises a `ValueError`. In `kmeans_50000_clustering.ipynb`, fitting K-Means seven times on 50,000 points for the elbow plot takes longer than the other cells.

## 13. Expected Output

### `7_k_means.ipynb`

- The 8-point DataFrame and the array of 4 centroids.
- After the input: `The point (x, y) belongs to Cluster <n>`.
- A scatter plot of all four clusters (red, green, blue, hotpink), the centroids as black `X` markers, and the new point as a yellow star with its cluster label.
- The saved outputs come from a re-run on 22 Nov 2026 with the inputs X = 6 and Y = 4 (the values in the earlier saved output), in which all 6 code cells ran:
  - Centroids `[[1.5, 3.5], [4.5, 8.5], [7, 4.333], [2, 10]]`.
  - `The point (6.0, 4.0) belongs to Cluster 2`, the cluster whose centroid is (7, 4.333).
  - The plot shows all four clusters.
- Because `KMeans` has no `random_state` here, a new run can number the clusters differently, so the cluster number printed for the same point can change.

### `kmeans_50000_clustering.ipynb`

This notebook has no saved outputs; it has not been run in its saved form. When run, it shows:

- The dataset shape, the first rows, missing-value counts, the number of duplicate `(X, Y)` rows and descriptive statistics.
- A scatter plot of the 50,000 original points.
- Elbow (inertia) and silhouette plots for K = 2 to 8, a table of both values, and `K with highest sampled silhouette score: <K>`.
- The 5 centroids, the number of points in each cluster and the first 10 rows with their cluster labels.
- `Inertia: <value>` and `Silhouette score on 5,000 sampled points: <value>`.
- A plot of the 5 clusters with centroids, then the same plot with the user's new point and its predicted cluster.
- `Saved clustered dataset to: kmeans_50000_with_clusters.csv` and `Saved trained model to: kmeans_model.joblib`.

## Changes (22 Nov 2026)

- `7_k_means.ipynb`: the plot loop drew only 3 of the 4 clusters (`range(3)` with three colours). It now runs `for i in range(kmeans.n_clusters)` with the colours `['red', 'green', 'blue', 'hotpink']`; `hotpink` comes from the author's own commented-out colour list. The notebook was re-run with X = 6 and Y = 4 (all 6 code cells ran) and its saved outputs were regenerated. The Windows username in the file path inside the saved warning output was replaced with `<user>`.
- `kmeans_50000_clustering.ipynb` and `kmeans_50000_data.csv`: unchanged.
