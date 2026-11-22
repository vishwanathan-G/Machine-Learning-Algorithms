# Support Vector Machine (SVM)

## Files in this folder

| File | Description |
|---|---|
| `5_svm_car_data.ipynb` | Multi-class SVM (linear kernel) that predicts car acceptability from six categorical attributes; reports accuracy, weighted F1 and a confusion matrix |
| `cars_dataset.csv` | 1,728 rows x 7 columns, all categorical |

## 1. Algorithm Name

Support Vector Machine, Support Vector Classifier (`SVC`), supervised learning, classification.

## 2. Definition

A Support Vector Machine finds the boundary (hyperplane) that separates classes with the widest possible margin. The training points closest to the boundary are called support vectors, and they alone determine where the boundary lies. Kernels let SVMs learn non-linear boundaries.

## 3. How It Works

1. Represent each sample as a point in feature space.
2. Find the hyperplane that maximises the margin, the distance to the nearest points of each class.
3. Allow some points inside the margin or on the wrong side (soft margin). The parameter `C` controls the penalty for these violations.
4. For non-linear problems, use a kernel function to compare points as if they had been mapped into a higher-dimensional space.
5. For more than two classes, `SVC` trains one-vs-one classifiers internally and combines their votes.

## 4. Why It Is Used

- It works well on medium-sized datasets with clear margins between classes.
- It is effective in high-dimensional spaces.
- Kernels make it flexible without changing the training algorithm.

## 5. Mathematical Intuition

- Decision function: `f(x) = w . x + b`. The predicted class is decided by the sign of `f(x)`.
- Margin width: `2 / ||w||`. Maximising the margin is the same as minimising `||w||`.
- Soft-margin objective: minimise `(1/2)||w||^2 + C * sum(xi_i)` subject to `y_i (w . x_i + b) >= 1 - xi_i` and `xi_i >= 0`.
- Kernel trick: replace `x_i . x_j` with `K(x_i, x_j)`. Common kernels:
  - linear: `x . x'`
  - polynomial: `(gamma * x . x' + r)^d`
  - RBF: `exp(-gamma * ||x - x'||^2)`

## 6. Important Parameters

**General (`sklearn.svm.SVC`)**

| Parameter | Default | Meaning |
|---|---|---|
| `C` | `1.0` | Penalty for margin violations; larger values fit the training data more tightly |
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `gamma` | `'scale'` | Kernel width for `rbf`, `poly`, `sigmoid` |
| `degree` | `3` | Degree for the `poly` kernel |
| `decision_function_shape` | `'ovr'` | Shape of `decision_function` output for multi-class. Training is always one-vs-one. |
| `class_weight` | `None` | `'balanced'` re-weights rare classes |

**Used in this notebook**

- `SVC(C=10.0, kernel="linear", decision_function_shape='ovr')`.
- Every feature column is label-encoded with `LabelEncoder` (the same encoder object is refitted for each column), then standardised with `StandardScaler`.
- `train_test_split(X, y, test_size=0.3, random_state=42)`: 1,209 training and 519 test rows.

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

- Text and document classification.
- Image classification and face detection.
- Bioinformatics, such as protein and gene classification.
- Quality and acceptability assessment from categorical ratings.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Loading the CSV |
| `numpy`, `matplotlib.pyplot` | Imported (not used directly) |
| `sklearn.preprocessing` (`LabelEncoder`, `StandardScaler`) | Encoding and scaling |
| `sklearn.model_selection.train_test_split` | Train/test split |
| `sklearn.svm.SVC` | The model |
| `sklearn.metrics` (`confusion_matrix`, `accuracy_score`, `f1_score`) | Evaluation |

## 11. Dataset Information

- File: `cars_dataset.csv`, 1,728 rows x 7 columns, no missing values.
- The data appears to be the UCI Car Evaluation dataset, with door and person counts written as words (`two`, `three`, `four`, `5more`; `two`, `four`, `more`).

| Column | Values |
|---|---|
| `buying` (price) | `vhigh`, `high`, `med`, `low` |
| `maint` (maintenance cost) | `vhigh`, `high`, `med`, `low` |
| `doors` | `two`, `three`, `four`, `5more` |
| `persons` (capacity) | `two`, `four`, `more` |
| `lug_boot` (boot size) | `small`, `med`, `big` |
| `safety` | `low`, `med`, `high` |
| `car` (target) | `unacc` 1,210, `acc` 384, `good` 69, `vgood` 65 |

**Change made for this repository.** With the author's approval, one line was changed. In the `pd.read_csv` call, the absolute local Windows path was replaced by the relative path `"cars_dataset.csv"`, so the notebook runs from this folder on any machine. No other code was changed, and the saved outputs are the author's original ones.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/SVM/5_svm_car_data.ipynb`.
3. Run the cells from top to bottom. There are no input prompts.

## 13. Expected Output

- The full DataFrame (1728 rows x 7 columns), the label-encoded feature matrix and the standardised feature matrix.
- The array of predicted labels for the 519 test rows.
- Training-set and test-set accuracy and weighted F1 score.
- A 4 x 4 confusion matrix. Rows and columns are in alphabetical label order: `acc`, `good`, `unacc`, `vgood`.

Saved output from the author's run:

| Metric | Training set | Test set |
|---|---|---|
| Accuracy | 0.7361 | 0.7168 |
| Weighted F1 | 0.6623 | 0.6341 |

Saved confusion matrix (test set; rows = actual, columns = predicted, order `acc`, `good`, `unacc`, `vgood`):

```
[[ 17   0 102   0]
 [  1   0  17   0]
 [  5   0 355   0]
 [ 12   0  10   0]]
```

In the saved run the model predicts only `unacc` (484 test rows) and `acc` (35 test rows); `good` and `vgood` are never predicted. About 69% of the test rows are `unacc`, so the accuracy is only a little above always predicting the majority class. A linear kernel on alphabetically label-encoded categories is a weak fit for this data. One-hot encoding or a non-linear kernel are common alternatives.

The notebook's first markdown cell is an "Open in Colab" badge and its second shows an image from an external web address. Both need an internet connection to display.
