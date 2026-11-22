# Principal Component Analysis (PCA)

## Files in this folder

| File | Description |
|---|---|
| `12_PCA.ipynb` | Explores the scikit-learn digits dataset, trains a logistic regression baseline on all 64 pixel features, then reduces dimensions with PCA (95% variance, and 2 components) and compares accuracy |

## Source and attribution

`12_PCA.ipynb` follows the codebasics "py" PCA tutorial ([github.com/codebasics/py](https://github.com/codebasics/py), folder `ML/18_PCA`, `PCA_tutorial_digits`). A comparison gave a code similarity score of 0.94, and 84% of the code lines are identical. The codebasics/py repository has no licence file.

## 1. Algorithm Name

Principal Component Analysis (PCA), unsupervised learning, dimensionality reduction.

## 2. Definition

PCA transforms a set of possibly correlated features into a smaller set of new, uncorrelated features called principal components. Each component is a linear combination of the original features. The components are ordered so that the first captures the most variance in the data, the second the next most, and so on.

## 3. How It Works

1. Centre the data by subtracting each feature's mean. Scaling to unit variance is optional but common.
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
- Explained variance ratio of component `k`: `lambda_k / sum(lambda)`. With `PCA(0.95)`, scikit-learn keeps the smallest number of components whose ratios add up to at least 0.95.
- The projected data is `Z = X V_k`, where `V_k` holds the first `k` eigenvectors as columns.

## 6. Important Parameters

**General (`sklearn.decomposition.PCA`)**

| Parameter | Default | Meaning |
|---|---|---|
| `n_components` | `None` | Integer: number of components. Float between 0 and 1: fraction of variance to keep. `None`: keep all. |
| `whiten` | `False` | Scale components to unit variance |
| `svd_solver` | `'auto'` | SVD algorithm |
| `random_state` | `None` | Seed for randomised solvers |

**Used in this notebook**

- Baseline: `StandardScaler()` on all 64 features, then `train_test_split(X_scaled, y, test_size=0.2, random_state=48)` and `LogisticRegression()` with defaults.
- `PCA(0.95)` on the unscaled features, then `train_test_split(X_pca, y, test_size=0.2, random_state=30)` and `LogisticRegression(max_iter=1000)`.
- `PCA(n_components=2)` on the unscaled features, with the same split and model settings.

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
- Pre-processing high-dimensional data, such as gene expression or sensor readings, before modelling.
- Visualising customer or document embeddings in 2D.
- Noise reduction in signals.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `sklearn.datasets.load_digits` | Built-in dataset |
| `pandas` | DataFrames for the features and the transformed data |
| `matplotlib.pyplot` | Showing digit images with `matshow` |
| `sklearn.preprocessing.StandardScaler` | Scaling for the baseline model |
| `sklearn.model_selection.train_test_split` | Train/test splits |
| `sklearn.linear_model.LogisticRegression` | Classifier used to compare feature sets |
| `sklearn.decomposition.PCA` | Dimensionality reduction |

## 11. Dataset Information

- Built-in scikit-learn digits dataset, loaded with `load_digits()`. No file is needed.
- 1,797 images of handwritten digits, each 8 x 8 pixels, flattened to 64 features (`pixel_0_0` to `pixel_7_7`). Pixel values range from 0 to 16.
- Target: the digit, 0 to 9.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `02_Unsupervised_Learning/PCA/12_PCA.ipynb`.
3. Run the cells from top to bottom.
4. The last code cell is a separate practice snippet, not part of the PCA workflow. It asks `enter num:`. Type a whole number such as `6`, and it prints a triangle of `*` characters: an empty line, then rows of 1 to `n - 1` stars.

## 13. Expected Output

- Dataset keys, the data shape `(1797, 64)`, the first image as a 64-value array and as an 8 x 8 array, and two digit images (samples 0 and 9) drawn with `matshow`.
- The feature DataFrame, `describe()` statistics, the scaled feature table and the accuracy of each model.

Saved output from the author's run:

| Step | Saved output |
|---|---|
| Logistic regression on 64 scaled features | accuracy 0.9583 |
| `PCA(0.95)` | 29 components (from 64); shape `(1797, 29)` |
| Logistic regression on 29 PCA components | accuracy 0.9694 |
| `PCA(n_components=2)` explained variance ratio | 0.1489 and 0.1362 (about 28% together) |
| Logistic regression on 2 PCA components | accuracy 0.6083 |
| Star-pattern cell | An empty line, then rows of 1 to 5 stars (consistent with an input of 6) |

With 29 components (95% of the variance) the classifier does about as well as with all 64 features. With only 2 components, about 28% of the variance, accuracy drops sharply. The baseline and the PCA models use different `random_state` values for their splits, and the baseline uses scaled features while PCA is fitted on unscaled ones, so the accuracies are not an exact like-for-like comparison.

## Changes (22 Nov 2026)

- `12_PCA.ipynb`: unchanged.
- This README: added the Source and attribution note.
