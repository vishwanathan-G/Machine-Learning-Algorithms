# K-Nearest Neighbors (KNN)

## Files in this folder

| File | Description |
|---|---|
| `Knn-1.ipynb` | Classifies Iris flowers with KNN for k = 5 and k = 50, then compares accuracy and confusion-matrix heatmaps |
| `data/iris-flower-dataset.csv` | The original Iris CSV used by the notebook, 150 rows x 5 columns |

## 1. Algorithm Name

K-Nearest Neighbors (KNN) classifier, supervised learning, classification.

## 2. Definition

KNN classifies a new point by looking at the `k` training points closest to it and taking a majority vote of their labels. It is an instance-based ("lazy") learner: it stores the training data and does the work at prediction time.

## 3. How It Works

1. Store all training points and their labels.
2. For a new point, compute its distance to every training point (Euclidean distance by default).
3. Select the `k` nearest points.
4. Predict the most common label among them. With `weights='distance'`, closer points count more.

## 4. Why It Is Used

- It is one of the simplest classifiers to understand and implement.
- There is no training phase, and decision boundaries of any shape are possible.
- It is a useful baseline and a clear way to see how the choice of `k` changes bias and variance.

## 5. Mathematical Intuition

- Euclidean distance: `d(a, b) = sqrt(sum((a_i - b_i)^2))`. This is Minkowski distance with `p = 2`. Manhattan distance is `p = 1`.
- A small `k` follows the training data closely (low bias, high variance, sensitive to noise). A large `k` smooths the boundary (higher bias, lower variance).
- When `k` approaches the size of the training set, the prediction tends toward the majority class.
- Because distances combine all features, features with larger ranges dominate unless the data is scaled.

## 6. Important Parameters

**General (`sklearn.neighbors.KNeighborsClassifier`)**

| Parameter | Default | Meaning |
|---|---|---|
| `n_neighbors` | `5` | Number of neighbours `k` |
| `weights` | `'uniform'` | `'uniform'` (equal votes) or `'distance'` (closer neighbours count more) |
| `metric` | `'minkowski'` | Distance function |
| `p` | `2` | Minkowski power: 2 = Euclidean, 1 = Manhattan |
| `algorithm` | `'auto'` | Neighbour search structure (`'ball_tree'`, `'kd_tree'`, `'brute'`) |

**Used in this notebook**

- `train_test_split(X, y, test_size=0.2, random_state=42)`: 120 training and 30 test rows.
- `KNeighborsClassifier(n_neighbors=5)` and `KNeighborsClassifier(n_neighbors=50)`.
- All four measurements are used as features without scaling. They are all in centimetres.

## 7. Advantages

- Simple and intuitive.
- No training time; new data can be added just by storing it.
- Naturally handles multi-class problems.
- Makes no assumption about the shape of the decision boundary.

## 8. Disadvantages

- Prediction is slow on large datasets, because distances to many points must be computed.
- Needs feature scaling when features have different units or ranges.
- Performance drops in high dimensions (the curse of dimensionality).
- Sensitive to the choice of `k`, to irrelevant features and to class imbalance.

## 9. Real-World Applications

- Recommendation systems ("users similar to you liked ...").
- Pattern and handwriting recognition.
- Anomaly detection based on distance to neighbours.
- Medical classification from a small number of measurements.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Loading the CSV |
| `numpy` | Imported (not used directly) |
| `matplotlib.pyplot` | Plotting backend for the heatmaps |
| `seaborn` | Confusion-matrix heatmaps |
| `sklearn.model_selection.train_test_split` | Train/test split |
| `sklearn.neighbors.KNeighborsClassifier` | The model |
| `sklearn.metrics` (`accuracy_score`, `confusion_matrix`) | Evaluation |

## 11. Dataset Information

- File: `data/iris-flower-dataset.csv`, 150 data rows x 5 columns.
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (cm). Target: `species`, with the labels `Iris-setosa`, `Iris-versicolor` and `Iris-virginica`, 50 rows each.
- This is the classic Iris flower dataset, also available from the UCI Machine Learning Repository.
- The file is the original dataset used by the notebook. It was recovered from the author's own files (an `IRIS.csv`), and re-running the notebook with it reproduces all of its saved outputs exactly (see section 13).

The notebook's first markdown cell shows an image from `media/iris-flower.png`. That image is not included in this repository, so it does not display.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/KNN/Knn-1.ipynb`. The notebook reads `data/iris-flower-dataset.csv` relative to its own folder.
3. Run the cells from top to bottom. There are no input prompts.

## 13. Expected Output

- The first five rows of the data, the feature matrix and the target.
- The predicted labels (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) for the 30 test rows for k = 5 and for k = 50.
- `Accuracy: <value>` for each model.
- Two confusion matrices (3 x 3) printed and drawn as seaborn heatmaps.

Saved output from the author's run. A re-run on 22 Nov 2026 with `data/iris-flower-dataset.csv` reproduced all 13 saved code-cell outputs exactly:

| Model | Accuracy | Confusion matrix (rows = actual Iris-setosa, Iris-versicolor, Iris-virginica) |
|---|---|---|
| k = 5 | 1.0 | `[[10 0 0] [0 9 0] [0 0 11]]` |
| k = 50 | 0.9666666666666667 | `[[10 0 0] [0 9 0] [0 1 10]]`: one Iris-virginica predicted as Iris-versicolor |

The comparison shows the bias-variance effect of `k`. With k = 50 each vote covers almost half of the 120 training points, so the boundary between the two overlapping classes becomes smoother and one test point is misclassified.

## Changes (22 Nov 2026)

- `data/iris-flower-dataset.csv`: the earlier substitute copy of `iris.csv` was replaced by the original dataset, recovered from the author's own files (150 rows; labels `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`).
- `Knn-1.ipynb`: unchanged. A re-run with the recovered file reproduced all 13 saved code-cell outputs exactly.
- This README: removed the note about the substitute copy.
