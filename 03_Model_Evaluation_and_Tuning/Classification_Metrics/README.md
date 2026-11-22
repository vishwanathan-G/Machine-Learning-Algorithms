# Classification Metrics

## Files in this folder

| File | Description |
|---|---|
| `11_evalution-1.ipynb` | Computes a confusion matrix, precision, recall and F1 score with scikit-learn for a small hand-made set of true and predicted labels, explains each formula in markdown, and draws the confusion matrix as a heatmap |

## 1. Algorithm Name

Classification evaluation metrics: confusion matrix, accuracy, precision, recall, F1 score.

## 2. Definition

Classification metrics measure how well a classifier's predicted labels match the true labels. The confusion matrix counts each kind of correct and incorrect prediction. Precision, recall and F1 summarise those counts from different points of view.

## 3. How It Works

For a binary problem with a positive class (1, "YES") and a negative class (0, "NO"), each prediction falls into one of four groups:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True Positive (TP) | False Negative (FN) |
| **Actually negative** | False Positive (FP) | True Negative (TN) |

The metrics are ratios of these counts. scikit-learn's `confusion_matrix` orders rows (actual) and columns (predicted) by sorted label value. For labels `[0, 1]` it returns `[[TN, FP], [FN, TP]]`, with the negative class first.

## 4. Why It Is Used

- Accuracy alone can be misleading, especially with imbalanced classes. A model that always predicts "NO" can be 95% accurate if only 5% of cases are "YES".
- Different problems care about different errors. Precision matters when false alarms are costly; recall matters when missed cases are costly.
- The confusion matrix shows exactly where the model goes wrong.

## 5. Mathematical Intuition

- Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
- Precision = `TP / (TP + FP)`: of everything predicted positive, how much was right.
- Recall (sensitivity) = `TP / (TP + FN)`: of all actual positives, how many were found.
- F1 = `2 * (Precision * Recall) / (Precision + Recall)`: the harmonic mean, which is high only when both are high.
- Specificity = `TN / (TN + FP)`: the recall of the negative class.
- Raising the decision threshold usually raises precision and lowers recall, and lowering it does the opposite.

## 6. Important Parameters

**General (`sklearn.metrics`)**

| Function | Parameter | Default | Meaning |
|---|---|---|---|
| `confusion_matrix` | `labels` | `None` | Order of classes in the matrix; sorted labels by default |
| `confusion_matrix` | `normalize` | `None` | `'true'`, `'pred'` or `'all'` to show proportions |
| `precision_score`, `recall_score`, `f1_score` | `pos_label` | `1` | Which label counts as positive |
| `precision_score`, `recall_score`, `f1_score` | `average` | `'binary'` | For multi-class: `'micro'`, `'macro'`, `'weighted'` |
| `precision_score`, `recall_score`, `f1_score` | `zero_division` | `'warn'` | Value to return when a denominator is 0 |

**Used in this notebook**

- All functions are called with their defaults: binary, positive label `1`.
- Heatmap: `sns.heatmap(cm, annot=True, fmt='d')`.

## 7. Advantages

- The confusion matrix gives a complete picture of the error types.
- Precision and recall map directly to business or clinical costs.
- F1 gives a single score that balances both and is more informative than accuracy on imbalanced data.

## 8. Disadvantages

- Precision, recall and F1 depend on the chosen decision threshold.
- F1 weights precision and recall equally, which may not match the real costs.
- Binary metrics need a choice of averaging method for multi-class problems.
- Accuracy hides class imbalance.

## 9. Real-World Applications

| Situation | Metric to prioritise |
|---|---|
| Spam filtering (avoid losing real email) | Precision |
| Disease screening (avoid missing patients) | Recall |
| Fraud detection (rare positives) | Recall, precision, F1 |
| Balanced need | F1 |

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `sklearn.metrics` (`confusion_matrix`, `precision_score`, `recall_score`, `f1_score`) | Metrics |
| `seaborn` | Confusion-matrix heatmap |
| `matplotlib.pyplot` | Axis labels and title for the heatmap |

## 11. Dataset Information

- Inline lists of 8 labels each, defined in the notebook. No file is needed:
  - `y_true = [1, 1, 0, 0, 1, 0, 1, 0]` (true labels, the "answer key")
  - `y_pred = [1, 0, 0, 0, 1, 1, 1, 0]` (predicted labels, the "answer sheet")
- Pair by pair this gives TP = 3, FN = 1, TN = 3, FP = 1. A comment in the notebook spells this out.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `03_Model_Evaluation_and_Tuning/Classification_Metrics/11_evalution-1.ipynb`.
3. Run the cells from top to bottom. There are no input prompts.

One cell runs `!pip install seaborn`. seaborn is already in `requirements.txt`, so you can skip that cell.

## 13. Expected Output

Saved output from the author's run:

| Cell | Saved output |
|---|---|
| `confusion_matrix(y_true, y_pred)` | `[[3 1] [1 3]]` |
| `precision_score` | 0.75 |
| `recall_score` | 0.75 |
| `f1_score` | 0.75 |

- Two identical heatmap cells each draw the confusion matrix with counts, labelled "Predicted" (x-axis) and "Actual" (y-axis).
- The markdown table in the notebook labels the first row "Actual YES" and the first column "Pred YES". scikit-learn's matrix puts class 0 ("NO") first. In this example the matrix is symmetric (`[[3 1] [1 3]]`), so the numbers match either way.
