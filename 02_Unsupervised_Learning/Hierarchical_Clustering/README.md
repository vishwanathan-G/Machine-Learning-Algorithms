# Hierarchical (Agglomerative) Clustering

## Files in this folder

| File | Description |
|---|---|
| `9_agg.ipynb` | Plots 15 inline points, draws a Ward-linkage dendrogram with SciPy, and clusters the points into 2 groups with scikit-learn's `AgglomerativeClustering` |

## 1. Algorithm Name

Hierarchical Clustering, agglomerative (bottom-up), unsupervised learning, clustering.

## 2. Definition

Agglomerative hierarchical clustering starts with every point as its own cluster and repeatedly merges the two closest clusters until one cluster remains. The sequence of merges forms a tree (hierarchy). A dendrogram draws this tree, and cutting it at a chosen height gives a flat set of clusters.

## 3. How It Works

1. Treat each of the `n` points as a cluster.
2. Compute the distance between every pair of clusters using a linkage rule.
3. Merge the closest pair.
4. Repeat steps 2 and 3 until a single cluster remains, recording the distance at each merge.
5. Draw the dendrogram, and cut it horizontally to get the desired number of clusters, or set `n_clusters` directly.

## 4. Why It Is Used

- No need to fix the number of clusters before seeing the structure; the dendrogram shows options at every level.
- The hierarchy itself is informative, for example in taxonomies or document topics.
- Results are deterministic for a given linkage and metric.

## 5. Mathematical Intuition

Linkage rules define the distance between clusters `A` and `B`:

| Linkage | Distance between clusters |
|---|---|
| Single | smallest distance between any point in A and any point in B |
| Complete | largest such distance |
| Average | mean of all pairwise distances |
| Ward (used here) | increase in total within-cluster sum of squares caused by merging A and B |

Ward linkage tends to produce compact, similarly sized clusters, similar in spirit to K-Means. In the dendrogram, a long vertical gap before a merge means two well-separated groups. Cutting through that gap gives a natural number of clusters.

## 6. Important Parameters

**General**

| Function / class | Parameter | Default | Meaning |
|---|---|---|---|
| `scipy.cluster.hierarchy.linkage` | `method` | `'single'` | Linkage rule (`'single'`, `'complete'`, `'average'`, `'ward'`, ...) |
| `scipy.cluster.hierarchy.linkage` | `metric` | `'euclidean'` | Distance between points |
| `sklearn.cluster.AgglomerativeClustering` | `n_clusters` | `2` | Number of clusters to return |
| `sklearn.cluster.AgglomerativeClustering` | `linkage` | `'ward'` | Linkage rule |
| `sklearn.cluster.AgglomerativeClustering` | `metric` | `'euclidean'` | Distance metric (Ward requires Euclidean) |
| `sklearn.cluster.AgglomerativeClustering` | `distance_threshold` | `None` | Cut height to use instead of `n_clusters` |

**Used in this notebook**

- `sch.linkage(data, method='ward')` for the dendrogram.
- `AgglomerativeClustering(n_clusters=2)`, which uses the default Ward linkage and Euclidean distance.

## 7. Advantages

- The dendrogram shows cluster structure at every scale.
- No random initialisation; results are reproducible.
- Works with many linkage rules and distance measures.
- Can find clusters that are not spherical with single or average linkage.

## 8. Disadvantages

- Time and memory grow at least with `n^2`, so it is impractical for very large datasets.
- Merges are greedy and cannot be undone.
- Sensitive to noise and outliers, especially with single linkage.
- The best cut height is still a judgement call.

## 9. Real-World Applications

- Building taxonomies, such as gene expression or species relationships.
- Grouping documents or news articles into topic hierarchies.
- Customer segmentation where nested segments are useful.
- Organising product catalogues.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Building the DataFrame |
| `matplotlib.pyplot` | Scatter plots and the dendrogram figure |
| `scipy.cluster.hierarchy` (`linkage`, `dendrogram`) | Dendrogram |
| `sklearn.cluster.AgglomerativeClustering` | Flat clustering |

## 11. Dataset Information

- Inline data defined in the notebook: 15 rows x 2 columns (`X`, `Y`).
- The points form three visible groups: 5 points near (1 to 5, 2 to 6), 5 near (8 to 12, 7 to 11) and 5 near (20 to 24, 20 to 24).
- No external file is needed.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started). SciPy is included in `requirements.txt`.
2. Run `python -m jupyter notebook` from the repository root and open `02_Unsupervised_Learning/Hierarchical_Clustering/9_agg.ipynb`.
3. Run the cells from top to bottom. There are no input prompts.

## 13. Expected Output

- The printed 15-row DataFrame.
- A scatter plot titled "Original Data".
- A dendrogram titled "Dendrogram" (x-axis "Points", y-axis "Distance").
- The array of cluster labels, and a scatter plot titled "Agglomerative Clustering Result" coloured by label.
- Saved output from the author's run: `[0 0 0 0 0 0 0 0 0 0 1 1 1 1 1]`. With 2 clusters, the two lower-left groups (the first 10 points) are merged into one cluster, and the far group (the last 5 points) forms the other.
