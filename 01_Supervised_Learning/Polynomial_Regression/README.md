# Polynomial Regression

## Files in this folder

| File | Description |
|---|---|
| `poly_reg.ipynb` | Compares plain linear regression with polynomial regression of degree 3 and 4 on inline experience/salary data, plots all fits and predicts a salary for a user-entered number of years |

## 1. Algorithm Name

Polynomial Regression, supervised learning, regression.

## 2. Definition

Polynomial regression fits a curve instead of a straight line. It expands the input `x` into powers (`x, x^2, ..., x^d`) and then fits an ordinary linear regression on those expanded features. The model is still linear in its coefficients, which is why it can be solved with `LinearRegression`.

## 3. How It Works

1. Choose a degree `d`.
2. Transform each input value into the feature vector `[1, x, x^2, ..., x^d]` (`PolynomialFeatures`).
3. Fit linear regression on the transformed features by least squares.
4. To predict, transform the new input with the same transformer and call `predict`.

## 4. Why It Is Used

- Many relationships curve: growth that speeds up, diminishing returns, saturation.
- It is an easy step up from linear regression and reuses the same fitting method.
- Comparing degrees on one plot shows clearly how model flexibility changes the fit.

## 5. Mathematical Intuition

- Degree-`d` model: `y_hat = b0 + b1*x + b2*x^2 + ... + bd*x^d`.
- The cost is the same as linear regression: `MSE = (1/n) * sum((y - y_hat)^2)`.
- A higher degree lowers the training error but increases variance. Beyond the training range the highest power dominates, so the curve can rise or fall steeply. This is the bias-variance trade-off.

## 6. Important Parameters

**General (`sklearn.preprocessing.PolynomialFeatures`)**

| Parameter | Default | Meaning |
|---|---|---|
| `degree` | `2` | Highest power to generate |
| `include_bias` | `True` | Add a column of ones (the `x^0` term) |
| `interaction_only` | `False` | With several features, keep only interaction terms such as `x1*x2` |

`LinearRegression` parameters are listed in [`../Linear_Regression/README.md`](../Linear_Regression/README.md#6-important-parameters).

**Used in this notebook**

- `LinearRegression()` on the raw `Experience` column (baseline).
- `PolynomialFeatures(degree=3)` followed by `LinearRegression()`.
- `PolynomialFeatures(degree=4)` followed by `LinearRegression()`.
- All models are trained on all 20 rows (no train/test split).
- Curves are drawn on a grid `np.arange(X.min(), X.max() + 0.1, 0.1)` so they look smooth.

## 7. Advantages

- Captures curved relationships with a simple, well-understood method.
- Reuses linear regression tooling and its closed-form solution.
- The degree is a single, easy-to-understand control over flexibility.

## 8. Disadvantages

- High degrees overfit and oscillate, especially with few data points.
- Extrapolation is unreliable: predictions outside the training range can grow very quickly.
- The number of features grows quickly when there are many input columns.
- Powers of large values differ greatly in scale; scaling often helps numerically.

## 9. Real-World Applications

- Salary or price curves that rise faster at higher levels.
- Growth curves in biology and population studies.
- Calibration curves for sensors and instruments.
- Modelling non-linear trends in physics and engineering measurements.

## 10. Python Libraries Used

| Library | Used for |
|---|---|
| `numpy` | Building the smooth plotting grid |
| `pandas` | Building the DataFrame |
| `matplotlib.pyplot` | 2 x 2 grid of plots |
| `sklearn.linear_model.LinearRegression` | Linear and polynomial fits |
| `sklearn.preprocessing.PolynomialFeatures` | Creating polynomial features |

## 11. Dataset Information

- Inline data defined in the notebook: 20 rows x 2 columns.
- `Experience`: 1 to 20 (years).
- `Salary`: 50,000 to 400,000. It rises slowly for the first few years and then by about 20,000 per year.
- No external file is needed.

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. Run `python -m jupyter notebook` from the repository root and open `01_Supervised_Learning/Polynomial_Regression/poly_reg.ipynb`.
3. Run the cells from top to bottom.
4. The last cell asks `Enter years of experience:`. Type a number such as `10` or `12.5` and press Enter. The training data covers 1 to 20 years, so values in that range show how the models interpolate. Values outside it show how they extrapolate. A non-numeric entry raises a `ValueError`.

## 13. Expected Output

- The 20-row DataFrame.
- A 2 x 2 figure of plots, each with the actual salaries as blue points:
  1. Linear regression (red line)
  2. Polynomial regression, degree 3 (green curve)
  3. Linear regression again (the third subplot repeats the first)
  4. Polynomial regression, degree 4 (purple curve)
- After the input, three printed predictions: linear, degree 3 and degree 4.
- Saved output from the author's run (input `25` years, outside the training range):

| Model | Saved prediction for 25 years |
|---|---|
| Linear | $468,784.21 |
| Polynomial, degree 3 | $479,299.21 |
| Polynomial, degree 4 | $630,984.31 |

The spread between the three values for an input beyond the data shows how differently the models extrapolate.
