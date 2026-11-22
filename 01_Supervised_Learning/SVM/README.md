# Support Vector Machine (SVM)

## Files in this folder

| File | Description |
|---|---|
| `svm_breast_cancer.ipynb` | Binary SVM classifier for the scikit-learn breast cancer dataset: effect of feature scaling, kernel comparison, `C`/`gamma` grid search, test-set evaluation and support vectors |

## Authorship

**`svm_breast_cancer.ipynb` was written by Claude (an AI assistant) on 22 November 2026, at the repository owner's request, as a reference implementation. It is not the owner's own work, and the owner has not yet reworked it.** The notebook's first cell says the same.

It replaces `5_svm_car_data.ipynb`, which was adapted from a public repository without a licence (see the root README's [Sources and Attribution](../../README.md#sources-and-attribution)). That notebook and its `cars_dataset.csv` were removed from this repository; they remain in the owner's local course folder.

## 1. Algorithm Name

Support Vector Machine, Support Vector Classifier (`SVC`), supervised learning, classification.

## 2. Definition

A Support Vector Machine finds the boundary (hyperplane) that separates classes with the widest possible margin. The training points closest to the boundary are called support vectors, and they alone determine where the boundary lies. Kernels let SVMs learn non-linear boundaries.

## 3. How It Works

1. Represent each sample as a point in feature space.
2. Find the hyperplane that maximises the margin, the distance to the nearest points of each class.
3. Allow some points inside the margin or on the wrong side (soft margin). The parameter `C` controls the penalty for these violations.
4. For non-linear problems, use a kernel function to compare points as if they had been mapped into a higher-dimensional space.
5. For more than two classes, `SVC` trains one-vs-one classifiers internally and combines their votes. (This notebook has two classes.)

## 4. Why It Is Used

- It works well on medium-sized datasets with clear margins between classes.
- It is effective in high-dimensional spaces, such as the 30 features used here.
- Kernels make it flexible without changing the training algorithm.

## 5. Mathematical Intuition

- Decision function: `f(x) = w . x + b`. The predicted class is decided by the sign of `f(x)`.
- Margin width: `2 / ||w||`. Maximising the margin is the same as minimising `||w||`.
- Soft-margin objective: minimise `(1/2)||w||^2 + C * sum(xi_i)` subject to `y_i (w . x_i + b) >= 1 - xi_i` and `xi_i >= 0`.
- Kernel trick: replace `x_i . x_j` with `K(x_i, x_j)`. Common kernels:
  - linear: `x . x'`
  - polynomial: `(gamma * x . x' + r)^d`
  - RBF: `exp(-gamma * ||x - x'||^2)`
- Because the RBF kernel depends on distances, features on large scales (here `mean area`, in the hundreds to thousands) dominate unless the data is standardised.

## 6. Important Parameters

**General (`sklearn.svm.SVC`)**

| Parameter | Default | Meaning |
|---|---|---|
| `C` | `1.0` | Penalty for margin violations; larger values fit the training data more tightly |
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `gamma` | `'scale'` | Kernel width for `rbf`, `poly`, `sigmoid`; larger values make the boundary more wiggly |
| `degree` | `3` | Degree for the `poly` kernel |
| `class_weight` | `None` | `'balanced'` re-weights rare classes |

**Used in this notebook**

- Every model is a pipeline `make_pipeline(StandardScaler(), SVC(...))`, so the scaler is fitted on training data only (one comparison model is deliberately left unscaled).
- `train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)`: 426 training and 143 test rows.
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Kernel comparison: `linear`, `poly`, `rbf` with default settings.
- Grid search over `C` in `[0.1, 1, 10, 100]` and `gamma` in `['scale', 0.001, 0.01, 0.1]` (RBF kernel, scoring `accuracy`).

## 7. Advantages

- Strong performance on many medium-sized problems.
- Only the support vectors matter, so the model is memory-efficient at prediction time.
- Flexible through kernels.
- The maximum-margin objective tends to generalise well.

## 8. Disadvantages

- Training time grows quickly with the number of samples.
- Sensitive to feature scaling and to the choice of `C`, `kernel` and `gamma`.
- Does not output probabilities directly (`probability=True` adds an extra, slower calibration step).
- Harder to interpret than linear or tree models, especially with non-linear kernels.

## 9. Real-World Applications

- Medical diagnosis from measurements, as in this notebook.
- Text and document classification.
- Image classification and face detection.
- Bioinformatics, such as protein and gene classification.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Holding the features and target |
| `numpy` | Imported |
| `matplotlib.pyplot` | Confusion-matrix plot |
| `sklearn.datasets.load_breast_cancer` | The dataset |
| `sklearn.model_selection` (`train_test_split`, `GridSearchCV`, `StratifiedKFold`, `cross_val_score`) | Splitting, cross-validation and tuning |
| `sklearn.preprocessing.StandardScaler`, `sklearn.pipeline.make_pipeline` | Scaling inside a pipeline |
| `sklearn.svm.SVC` | The model |
| `sklearn.metrics` (`accuracy_score`, `classification_report`, `ConfusionMatrixDisplay`) | Evaluation |

## 11. Dataset Information

- Built-in: `sklearn.datasets.load_breast_cancer()` (the Wisconsin Diagnostic Breast Cancer dataset). No file and no internet connection are needed.
- 569 rows x 30 numeric features computed from images of cell nuclei (mean, standard error and "worst" value of radius, texture, perimeter, area, smoothness and others).
- Target: `0` = malignant (212 rows), `1` = benign (357 rows).
- The features are on very different scales; for example `mean smoothness` is about 0.05 to 0.16, while `mean radius` is about 7 to 28.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/SVM/svm_breast_cancer.ipynb`.
3. Run the cells from top to bottom. There are no input prompts. The whole notebook runs in a few seconds on a laptop CPU.

## 13. Expected Output

Results from running the notebook on 22 Nov 2026 (Python 3.10.11, scikit-learn 1.7.2). They are saved in the notebook. Because the split and cross-validation folds use `random_state=42`, a re-run with the same library versions gives the same numbers.

**Effect of scaling** (RBF kernel, default settings, test set):

| Model | Test accuracy |
|---|---|
| RBF SVM without scaling | 0.923 |
| RBF SVM with `StandardScaler` | 0.979 |

**Kernel comparison** (5-fold cross-validation on the training set, scaled):

| Kernel | Mean CV accuracy | Std |
|---|---|---|
| `linear` | 0.972 | 0.012 |
| `poly` | 0.894 | 0.036 |
| `rbf` | 0.967 | 0.005 |

**Grid search** (RBF kernel): best parameters `C = 10`, `gamma = 0.001`, best mean CV accuracy 0.974. The notebook also prints the mean CV accuracy for all 16 `C`/`gamma` pairs; the weakest pair is `C = 0.1`, `gamma = 0.001` (0.706).

**Tuned model on the test set** (143 rows): accuracy 0.979.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| malignant | 0.98 | 0.96 | 0.97 | 53 |
| benign | 0.98 | 0.99 | 0.98 | 90 |

The notebook also draws the test-set confusion matrix.

**Support vectors:** 38 malignant and 40 benign, 78 of the 426 training points.

## Changes (22 Nov 2026)

- Added `svm_breast_cancer.ipynb`, a reference implementation written by Claude at the owner's request (see Authorship).
- Removed `5_svm_car_data.ipynb` and `cars_dataset.csv`, which were adapted from an unlicensed public repository. They remain in the owner's local course folder.
- This README was rewritten for the new notebook.
