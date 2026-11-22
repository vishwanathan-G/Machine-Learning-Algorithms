# Principal Component Analysis (PCA)

## Files in this folder

| File | Description |
|---|---|
| `pca_wine.ipynb` | PCA on the scikit-learn wine dataset: standardisation, explained variance, a 2-component scatter plot, component loadings, and a cross-validated comparison of logistic regression on all features vs. on principal components |

## Authorship

**`pca_wine.ipynb` was written by Claude (an AI assistant) on 22 November 2026, at the repository owner's request, as a reference implementation. It is not the owner's own work, and the owner has not yet reworked it.** The notebook's first cell says the same.

It replaces `12_PCA.ipynb`, which closely followed a public tutorial without a licence (see the root README's [Sources and Attribution](../../README.md#sources-and-attribution)). That notebook was removed from this repository; it remains in the owner's local course folder.

## 1. Algorithm Name

Principal Component Analysis (PCA), unsupervised learning, dimensionality reduction.

## 2. Definition

PCA transforms a set of possibly correlated features into a smaller set of new, uncorrelated features called principal components. Each component is a linear combination of the original features. The components are ordered so that the first captures the most variance in the data, the second the next most, and so on.

## 3. How It Works

1. Centre the data by subtracting each feature's mean. Scaling to unit variance is common, and is done here.
2. Find the directions of maximum variance: the eigenvectors of the covariance matrix, computed in practice with a singular value decomposition (SVD).
3. Sort the directions by how much variance they explain.
4. Keep the top `k` components and project the data onto them.

## 4. Why It Is Used

- To reduce the number of features, which speeds up training and can reduce overfitting.
- To remove redundancy between correlated features.
- To visualise high-dimensional data in 2 or 3 dimensions.
- To compress data and filter noise.

## 5. Mathematical Intuition

- For centred data `X` (n rows, d features), the covariance matrix is `C = (1 / (n - 1)) * X^T X`.
- Principal components are the eigenvectors `v` of `C` (`C v = lambda v`). Each eigenvalue `lambda` is the variance along its direction.
- Explained variance ratio of component `k`: `lambda_k / sum(lambda)`. The cumulative sum tells how much of the total variance the first `k` components keep.
- The projected data is `Z = X V_k`, where `V_k` holds the first `k` eigenvectors as columns. The entries of `V_k` are the loadings: how much each original feature contributes to each component.

## 6. Important Parameters

**General (`sklearn.decomposition.PCA`)**

| Parameter | Default | Meaning |
|---|---|---|
| `n_components` | `None` | Integer: number of components. Float between 0 and 1: fraction of variance to keep. `None`: keep all. |
| `whiten` | `False` | Scale components to unit variance |
| `svd_solver` | `'auto'` | SVD algorithm |
| `random_state` | `None` | Seed for randomised solvers |

**Used in this notebook**

- `StandardScaler()` on all 13 features before PCA.
- `PCA()` with all components to get the explained-variance table; `PCA(n_components=2)` for the scatter plot.
- The number of components for 95% of the variance is computed from the cumulative explained variance.
- Comparison models: `make_pipeline(StandardScaler(), [PCA(n_components=k)], LogisticRegression(max_iter=1000))`, scored with `cross_val_score` and `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Scaling and PCA are inside the pipeline, so they are re-fitted on each training fold.

## 7. Advantages

- Reduces dimensionality while keeping most of the information (variance).
- Removes correlation between features.
- Speeds up downstream models.
- Enables 2D or 3D visualisation of complex data.

## 8. Disadvantages

- Components are combinations of the original features, so they are harder to interpret.
- Captures only linear structure.
- Sensitive to feature scale. Features with large variance dominate unless the data is standardised.
- Maximum variance is not always the same as the most useful information for a prediction task.

## 9. Real-World Applications

- Image compression and face recognition ("eigenfaces").
- Pre-processing high-dimensional data, such as gene expression, sensor or chemical measurements, before modelling.
- Visualising customer or document embeddings in 2D.
- Noise reduction in signals.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `sklearn.datasets.load_wine` | The dataset |
| `numpy`, `pandas` | Arrays, tables of explained variance and loadings |
| `matplotlib.pyplot` | Explained-variance chart and 2D scatter plot |
| `sklearn.preprocessing.StandardScaler` | Standardisation |
| `sklearn.decomposition.PCA` | Dimensionality reduction |
| `sklearn.pipeline.make_pipeline` | Scaling, PCA and the classifier in one pipeline |
| `sklearn.linear_model.LogisticRegression` | Classifier used to compare feature sets |
| `sklearn.model_selection` (`cross_val_score`, `StratifiedKFold`) | Cross-validation |

## 11. Dataset Information

- Built-in: `sklearn.datasets.load_wine()` (the UCI Wine recognition dataset). No file and no internet connection are needed.
- 178 wines from three cultivars, 13 numeric features from chemical analysis (alcohol, malic acid, ash, flavanoids, colour intensity, proline and others).
- Classes: 0 (59 wines), 1 (71), 2 (48).
- The features are on very different scales; for example `proline` ranges from 278 to 1,680, while `hue` ranges from 0.48 to 1.71. This is why the data is standardised first.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `02_Unsupervised_Learning/PCA/pca_wine.ipynb`.
3. Run the cells from top to bottom. There are no input prompts. The notebook runs in a few seconds.

## 13. Expected Output

Results from running the notebook on 22 Nov 2026 (Python 3.10.11, scikit-learn 1.7.2), saved in the notebook. PCA is deterministic for this data, and the cross-validation folds use `random_state=42`, so a re-run with the same library versions gives the same numbers.

**Explained variance** (first components):

| Component | Explained variance | Cumulative |
|---|---|---|
| PC1 | 0.362 | 0.362 |
| PC2 | 0.192 | 0.554 |
| PC3 | 0.111 | 0.665 |
| PC4 | 0.071 | 0.736 |
| PC5 | 0.066 | 0.802 |

The full table lists all 13 components. **10 of the 13 components are needed to keep 95% of the variance.** The notebook also draws a bar chart of the explained variance with the cumulative line and a 95% reference line.

**2D projection:** a scatter plot of the wines on PC1 and PC2, coloured by cultivar.

**Largest loadings** (absolute values):

| PC1 | | PC2 | |
|---|---|---|---|
| flavanoids | 0.423 | color_intensity | 0.530 |
| total_phenols | 0.395 | alcohol | 0.484 |
| od280/od315_of_diluted_wines | 0.376 | proline | 0.365 |
| proanthocyanins | 0.313 | ash | 0.316 |

**Logistic regression with and without PCA** (5-fold cross-validation):

| Features used | Mean CV accuracy | Std |
|---|---|---|
| All 13 | 0.983 | 0.014 |
| 2 principal components | 0.955 | 0.022 |
| 10 principal components | 0.983 | 0.014 |

Two components keep about 55% of the variance and still give 0.955 accuracy; ten components match the full feature set.

## Changes (22 Nov 2026)

- Added `pca_wine.ipynb`, a reference implementation written by Claude at the owner's request (see Authorship).
- Removed `12_PCA.ipynb`, which closely followed an unlicensed public tutorial. It remains in the owner's local course folder.
- This README was rewritten for the new notebook.
