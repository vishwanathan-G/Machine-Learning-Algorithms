# Decision Tree

## Files in this folder

| File | Description |
|---|---|
| `4_decision tree.ipynb` | Trains a decision tree classifier on a 10-row inline categorical dataset (label-encoded), then classifies a user-entered combination of values |

## 1. Algorithm Name

Decision Tree (classification), supervised learning.

## 2. Definition

A decision tree is a model that predicts by asking a sequence of yes/no questions about the features. Each internal node tests one feature, each branch is an answer, and each leaf holds a predicted class.

## 3. How It Works

1. Start with all training rows at the root.
2. For every feature and candidate split point, measure how "pure" the resulting groups would be (Gini impurity or entropy).
3. Choose the split that reduces impurity the most and divide the rows.
4. Repeat on each child node until the nodes are pure or a stopping rule is reached (maximum depth, minimum samples, and so on).
5. To predict, follow the questions from the root to a leaf and return that leaf's majority class.

scikit-learn implements an optimised version of CART, which always makes binary splits.

## 4. Why It Is Used

- The model is a set of readable if/else rules.
- It handles non-linear relationships and feature interactions without feature scaling.
- It is the building block of Random Forests and gradient-boosted trees.

## 5. Mathematical Intuition

- Gini impurity of a node: `Gini = 1 - sum(p_k^2)`, where `p_k` is the share of class `k` in the node. 0 means pure.
- Entropy: `H = -sum(p_k * log2(p_k))`.
- Information gain of a split: `IG = impurity(parent) - sum((n_child / n_parent) * impurity(child))`.
- The tree greedily chooses the split with the largest impurity reduction at each step.

## 6. Important Parameters

**General (`sklearn.tree.DecisionTreeClassifier`)**

| Parameter | Default | Meaning |
|---|---|---|
| `criterion` | `'gini'` | Impurity measure (`'gini'`, `'entropy'`, `'log_loss'`) |
| `max_depth` | `None` | Maximum depth; `None` grows until leaves are pure |
| `min_samples_split` | `2` | Minimum rows needed to split a node |
| `min_samples_leaf` | `1` | Minimum rows in a leaf |
| `max_features` | `None` | Number of features considered per split |
| `random_state` | `None` | Seed for tie-breaking between equally good splits |

**Used in this notebook**

- `DecisionTreeClassifier()` with all defaults (Gini, unlimited depth).
- Four separate `LabelEncoder` objects: one each for `a1`, `a2`, `a3` and the target `Classification`.
- Trained on all 10 rows (no train/test split).

## 7. Advantages

- Easy to visualise and explain.
- Works with numeric and encoded categorical features; no scaling needed.
- Captures interactions between features automatically.
- Fast to predict.

## 8. Disadvantages

- Overfits easily when grown to full depth.
- Small changes in the data can produce a very different tree (high variance).
- Splits are axis-aligned, so diagonal boundaries need many splits.
- Greedy splitting does not guarantee the best overall tree.

## 9. Real-World Applications

- Loan approval and credit rules.
- Medical triage and diagnosis checklists.
- Customer segmentation and churn rules.
- Simple game-playing or weather-style "should we play?" decisions.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `pandas` | Building the DataFrame |
| `sklearn.preprocessing.LabelEncoder` | Converting text categories to integers |
| `sklearn.tree.DecisionTreeClassifier` | The model |

## 11. Dataset Information

Inline data defined in the notebook: 10 rows x 4 columns, all text values.

| Column | Values | Encoded as (`LabelEncoder`, alphabetical) |
|---|---|---|
| `a1` | `True`, `False` | False = 0, True = 1 |
| `a2` | `Hot`, `Cool` | Cool = 0, Hot = 1 |
| `a3` | `High`, `Normal` | High = 0, Normal = 1 |
| `Classification` (target) | `Yes` (6 rows), `No` (4 rows) | No = 0, Yes = 1 |

In this data every row with `a1 = False` is `Yes`. Rows with `a1 = True` are `Yes` only when `a3 = Normal`.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/Decision_Tree/4_decision tree.ipynb`.
3. Run the cells from top to bottom. The input cell asks three questions. Type the value exactly as shown, because the case must match, and press Enter after each:

| Prompt | Type one of |
|---|---|
| `Enter value for a1 (True/False):` | `True` or `False` |
| `Enter value for a2 (Hot/Cool):` | `Hot` or `Cool` |
| `Enter value for a3 (High/Normal):` | `High` or `Normal` |

Any other spelling (for example `true` or `hot`) is caught, and the cell prints `Invalid input. Please enter correct values ...`.

## 13. Expected Output

- The original 10-row DataFrame, then the same DataFrame with the encoded columns `a1_enc`, `a2_enc`, `a3_enc` and `target` added.
- The fitted model `DecisionTreeClassifier()`.
- After the three inputs: `Classification -> Yes` or `Classification -> No`.
- Saved output from the author's run: `Classification -> Yes` (the three values typed were not recorded).
- A scikit-learn `UserWarning` about missing feature names may appear, because the prediction is made from a plain list.
