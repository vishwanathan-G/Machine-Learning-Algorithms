# Random Forest

## Files in this folder

| File | Description |
|---|---|
| `Random_forest.ipynb` | Random Forest classifier on the scikit-learn Wine dataset with a stratified split, classification report, confusion matrix, feature importances and two plots. The notebook has no saved outputs. |

## 1. Algorithm Name

Random Forest (classification), supervised learning, ensemble method.

## 2. Definition

A Random Forest is an ensemble of decision trees. Each tree is trained on a random bootstrap sample of the rows and considers a random subset of features at each split. The forest predicts the class that the trees vote for most often.

## 3. How It Works

1. Draw a bootstrap sample (rows sampled with replacement) from the training data for each tree.
2. Grow a decision tree on that sample. At every split, consider only a random subset of the features.
3. Repeat for `n_estimators` trees.
4. To predict, let every tree vote. scikit-learn averages the trees' class probabilities and picks the highest.
5. Feature importance is the average impurity decrease that each feature produces across all trees.

## 4. Why It Is Used

- It is usually much more accurate and stable than a single decision tree.
- It works well with little tuning and no feature scaling.
- It gives feature importances that help explain which inputs matter.

## 5. Mathematical Intuition

- Averaging many models with low bias but high variance reduces variance. If the trees' errors are not perfectly correlated, the ensemble error is lower than a typical single tree's.
- Random feature selection at each split de-correlates the trees, which makes the averaging more effective.
- Each bootstrap sample contains about 63.2% of the distinct rows (`1 - 1/e`). The remaining "out-of-bag" rows can be used for validation.
- Prediction: `y_hat = argmax_k (1/T) * sum_t P_t(class k | x)` over `T` trees.

## 6. Important Parameters

**General (`sklearn.ensemble.RandomForestClassifier`)**

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | `100` | Number of trees |
| `max_depth` | `None` | Maximum depth of each tree |
| `max_features` | `'sqrt'` | Features considered at each split |
| `min_samples_split` / `min_samples_leaf` | `2` / `1` | Minimum rows to split / in a leaf |
| `bootstrap` | `True` | Sample rows with replacement for each tree |
| `class_weight` | `None` | `'balanced'` weights classes inversely to their frequency |
| `random_state` | `None` | Seed for reproducibility |
| `n_jobs` | `None` | CPU cores used to build trees in parallel |

**Used in this notebook**

- `RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")`.
- `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`: 80/20 split that keeps the class proportions.

## 7. Advantages

- High accuracy on tabular data with little tuning.
- Robust to overfitting compared with a single deep tree.
- Handles non-linear relationships and feature interactions; no scaling needed.
- Provides feature importances.
- Trees can be built in parallel.

## 8. Disadvantages

- Less interpretable than a single tree.
- Larger models need more memory and are slower to predict than simple models.
- Impurity-based feature importance can favour features with many distinct values.
- Does not extrapolate beyond the range of the training targets in regression.

## 9. Real-World Applications

- Credit scoring and fraud detection.
- Medical diagnosis from lab measurements.
- Product quality classification, such as wine or food grading from chemical analysis.
- Customer churn and marketing response prediction.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `sklearn.datasets.load_wine` | Built-in dataset |
| `sklearn.ensemble.RandomForestClassifier` | The model |
| `sklearn.model_selection.train_test_split` | Stratified split |
| `sklearn.metrics` (`accuracy_score`, `classification_report`, `confusion_matrix`) | Evaluation |
| `matplotlib.pyplot`, `seaborn` | Confusion-matrix heatmap and feature-importance bar chart |
| `pandas` | Sorting feature importances as a Series |

## 11. Dataset Information

- Built-in scikit-learn Wine dataset, loaded with `load_wine(as_frame=True)`. No file is needed.
- 178 rows x 13 numeric features from chemical analysis (for example `alcohol`, `malic_acid`, `flavanoids`, `color_intensity`, `proline`).
- Target: 3 cultivar classes, named `class_0`, `class_1`, `class_2`.
- After the split: 142 training rows and 36 test rows.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/Random_Forest/Random_forest.ipynb`.
3. Run the cells from top to bottom. There are no input prompts.

## 13. Expected Output

This notebook has no saved outputs; it has not been run in its saved form. When run, it prints and plots the following:

- `Accuracy: <value>` to three decimals.
- A classification report with precision, recall, F1 and support for `class_0`, `class_1`, `class_2`.
- A 3 x 3 confusion matrix printed as an array.
- `Feature importance:`, followed by all 13 features sorted from most to least important.
- `Predicted wine class: <class name>` for the first test row.
- Two plots: a blue confusion-matrix heatmap with class names on the axes, and a horizontal bar chart of the 10 most important features.

Because `random_state=42` is fixed for both the split and the forest, repeated runs in the same environment should print the same numbers.
