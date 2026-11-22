# DBSCAN

## Files in this folder

| File | Description |
|---|---|
| `10_dbscan.ipynb` | Short example: DBSCAN on a generated two-moons dataset, with plots before and after clustering. The notebook has no saved outputs. |
| `DBSCAN_Clustering_Practical.ipynb` | Longer guided version: the same idea plus cluster and noise counts, a results table, a comparison of three `eps` values and a k-distance graph. The notebook has no saved outputs. |

## 1. Algorithm Name

DBSCAN (Density-Based Spatial Clustering of Applications with Noise), unsupervised learning, clustering.

## 2. Definition

DBSCAN groups together points that lie in dense regions and marks points in sparse regions as noise. A cluster is a connected area where every point has enough neighbours within a given radius. The number of clusters does not need to be specified.

## 3. How It Works

1. For each point, count the neighbours within distance `eps`. The count includes the point itself.
2. A point with at least `min_samples` neighbours is a **core point**.
3. Core points that are within `eps` of each other are connected into the same cluster. The cluster grows through chains of core points.
4. A non-core point within `eps` of a core point is a **border point** and joins that cluster.
5. Points that are neither core nor border are **noise** and get the label `-1`.

## 4. Why It Is Used

- It finds clusters of arbitrary shape, such as the two interlocking moons here, which K-Means cannot separate.
- It identifies outliers explicitly.
- The number of clusters does not need to be known in advance.

## 5. Mathematical Intuition

- Epsilon-neighbourhood of a point `p`: `N_eps(p) = { q : dist(p, q) <= eps }`.
- `p` is a core point if `|N_eps(p)| >= min_samples`.
- `q` is density-reachable from `p` if a chain of core points links them, each within `eps` of the next. A cluster is a maximal set of density-connected points.
- Choosing `eps`: sort every point's distance to its k-th nearest neighbour (k close to `min_samples`) and look for the "knee" where distances rise sharply. Points beyond the knee are likely noise.

## 6. Important Parameters

**General (`sklearn.cluster.DBSCAN`)**

| Parameter | Default | Meaning |
|---|---|---|
| `eps` | `0.5` | Neighbourhood radius |
| `min_samples` | `5` | Points within `eps`, including the point itself, needed for a core point |
| `metric` | `'euclidean'` | Distance metric |
| `algorithm` | `'auto'` | Neighbour search method |

**Used in these notebooks**

- Data: `make_moons(n_samples=300, noise=0.05)`. `DBSCAN_Clustering_Practical.ipynb` adds `random_state=42`. `10_dbscan.ipynb` does not, so its data changes slightly on each run.
- Model: `DBSCAN(eps=0.2, min_samples=5)` in both notebooks.
- Parameter comparison (practical notebook only): `eps` = 0.1, 0.2 and 0.3, each with `min_samples=5`.
- K-distance graph (practical notebook only): `NearestNeighbors(n_neighbors=5)`.

## 7. Advantages

- Finds clusters of any shape.
- Detects noise and outliers automatically.
- No need to choose the number of clusters.
- Only two main parameters.

## 8. Disadvantages

- Results are sensitive to `eps` and `min_samples`.
- Struggles when clusters have very different densities, because one `eps` cannot fit all of them.
- Distances become less meaningful in high dimensions.
- Needs feature scaling when features have different units.

## 9. Real-World Applications

- Geographic clustering of GPS points, such as hotspots or delivery zones.
- Anomaly and fraud detection, where noise points are the anomalies.
- Grouping points in image or LiDAR point-cloud data.
- Finding dense groups in sensor or network traffic data.

## 10. Python Libraries Used

| Library | Used for | Notebooks |
|---|---|---|
| `numpy` | Counting clusters and noise, sorting distances | `DBSCAN_Clustering_Practical` (imported but not used in `10_dbscan`) |
| `pandas` | Results table | `DBSCAN_Clustering_Practical` (imported but not used in `10_dbscan`) |
| `matplotlib.pyplot` | Scatter plots and the k-distance graph | both |
| `sklearn.cluster.DBSCAN` | The model | both |
| `sklearn.datasets.make_moons` | Generated data | both |
| `sklearn.neighbors.NearestNeighbors` | K-distance graph | `DBSCAN_Clustering_Practical` |

## 11. Dataset Information

- Generated data from `sklearn.datasets.make_moons`: 300 points x 2 features in two interleaving half-circles, with small Gaussian noise (`noise=0.05`).
- `make_moons` also returns true moon labels (0 or 1). The practical notebook uses them only to colour the first plot and as a reference column in the results table. DBSCAN never sees them.
- No external file is needed.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open a notebook in `02_Unsupervised_Learning/DBSCAN/`.
3. Run the cells from top to bottom. There are no input prompts.

`DBSCAN_Clustering_Practical.ipynb` starts with an optional `%pip install numpy pandas matplotlib scikit-learn` cell. Skip it if you have already installed `requirements.txt`.

## 13. Expected Output

Neither notebook has saved outputs; neither has been run in its saved form. When run, they show:

### `10_dbscan.ipynb`

- A scatter plot titled "Original Data" (two moon shapes).
- A scatter plot titled "DBSCAN Clustering", coloured by the labels DBSCAN found (`rainbow` colour map). Noise points, if any, get their own colour.

### `DBSCAN_Clustering_Practical.ipynb`

- `Libraries imported successfully.`
- A plot of the original two-moons data coloured by the true labels, then `Dataset shape: (300, 2)` and `Number of data points: 300`.
- `Cluster labels found: [...]`, `Number of clusters (excluding noise): <n>` and `Number of noise points: <n>`.
- A scatter plot titled "DBSCAN Clustering (eps=0.2, min_samples=5)".
- The first 10 rows of a results table (`Feature_1`, `Feature_2`, `Cluster`, `Actual_Label_For_Reference`) and a count of points per cluster, where `-1` is noise.
- Three plots side by side for `eps` = 0.1, 0.2 and 0.3. Each title shows the cluster count and the noise count.
- A k-distance graph (k = 5), sorted distance against point index, for choosing `eps`. `kneighbors` is called on the training points themselves, so each point counts as its own first neighbour. The plotted value is therefore the distance to the 4th other point, which matches how `min_samples` counts the point itself.
