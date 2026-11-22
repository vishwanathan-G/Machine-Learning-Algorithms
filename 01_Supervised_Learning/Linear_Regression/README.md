# Linear Regression

## Files in this folder

| File | Description |
|---|---|
| `Linear Regression_Delivery Time Prediction.ipynb` | Simple linear regression: delivery time from distance, R2 on a hold-out split, a plot of the fitted line, and saving and re-loading the model with joblib; asks for a distance twice |
| `Multiple Linear Regression__House Price Prediction.ipynb` | Multiple linear regression: house price from area, bedrooms, house age and distance to the city centre, R2 on a hold-out split and an actual-vs-predicted plot; asks for four values |
| `1_1linear_regression.ipynb` | Multiple linear regression on 50,000 rows with an 80/20 train/test split and MAE, MSE, RMSE, R2 |
| `multi_linear_reg.ipynb` | Multiple linear regression for CO2 emissions from inline vehicle data; asks for input |
| `01_linear_delivery_time.csv` | 250 rows x 2 columns (`Distance_km`, `Delivery_Time_min`) |
| `02_multiple_linear_house_price.csv` | 250 rows x 5 columns (`Area_sqft`, `Bedrooms`, `House_Age_years`, `Distance_to_CityCenter_km`, `Price_Lakhs`) |
| `homeprices.csv` | 50,000 rows x 4 columns (area, bedrooms, age, price) |

## Source

All four notebooks are the author's own work. `Linear Regression_Delivery Time Prediction.ipynb` and `Multiple Linear Regression__House Price Prediction.ipynb` come from the author's own practice repository (`vishwanathan-G/ML_project`), where they were written as solutions to a set of regression practice questions; `01_linear_delivery_time.csv` and `02_multiple_linear_house_price.csv` are the practice datasets supplied with those questions. They replaced two notebooks that closely followed the codebasics tutorials (see [Changes](#changes-22-nov-2026)). A comparison with the tutorial sources gave code similarity scores of 0.30 to 0.37 for the replacement notebooks. `1_1linear_regression.ipynb` and `multi_linear_reg.ipynb` did not match those sources.

## 1. Algorithm Name

Linear Regression (simple and multiple), supervised learning, regression.

## 2. Definition

Linear regression models a continuous target as a weighted sum of input features plus a constant. Simple linear regression uses one feature. Multiple (multivariate) linear regression uses two or more.

## 3. How It Works

1. Assume the target is roughly a straight-line (or flat-plane) function of the features.
2. Choose the weights (coefficients) and the intercept that make the predicted values as close as possible to the actual values, by minimising the sum of squared errors (ordinary least squares).
3. Use the fitted equation to predict the target for new inputs.

## 4. Why It Is Used

- It is the standard first model for a numeric target and a baseline for more complex models.
- The coefficients are directly interpretable: each one is the change in the prediction for a one-unit change in that feature, holding the others fixed.
- It trains almost instantly, even on large tables.

## 5. Mathematical Intuition

- Simple: `y_hat = m * x + b`, where `m` is the slope (`coef_`) and `b` is the intercept (`intercept_`).
- Multiple: `y_hat = b0 + b1*x1 + b2*x2 + ... + bn*xn`.
- Cost function (mean squared error): `MSE = (1/n) * sum((y - y_hat)^2)`.
- Closed-form least-squares solution: `w = (X^T X)^-1 X^T y`. Gradient descent reaches the same minimum iteratively.
- Common metrics:
  - `MAE = mean(|y - y_hat|)`
  - `RMSE = sqrt(MSE)`, in the same units as the target
  - `R2 = 1 - SS_res / SS_tot`, the share of the target's variance the model explains (1.0 is a perfect fit)

## 6. Important Parameters

**General (`sklearn.linear_model.LinearRegression`)**

| Parameter | Default | Meaning |
|---|---|---|
| `fit_intercept` | `True` | Whether to learn the constant term `b` |
| `positive` | `False` | Force all coefficients to be non-negative |
| `copy_X` | `True` | Copy the input instead of overwriting it |
| `n_jobs` | `None` | Number of CPU cores for large multi-target problems |

**Used in these notebooks**

- `LinearRegression()` with all defaults in every notebook (written as `linear_model.LinearRegression()` in the Delivery Time and House Price notebooks).
- `Linear Regression_Delivery Time Prediction.ipynb`: feature `Distance_km`, target `Delivery_Time_min`. `train_test_split(x, y, train_size=0.8)` gives 200 training and 50 test rows. No `random_state` is set, so every run uses a different split and gives a different R2 and different predictions. The fitted model is saved with `joblib.dump(model, "delivery_model.pkl")` and loaded again with `joblib.load("delivery_model.pkl")`.
- `Multiple Linear Regression__House Price Prediction.ipynb`: features `Area_sqft`, `Bedrooms`, `House_Age_years`, `Distance_to_CityCenter_km` (every column except the target), target `Price_Lakhs`. `train_test_split(x, y, train_size=0.8)`, also without `random_state`.
- The Delivery Time and House Price notebooks evaluate the model with `r2_score` on the test set and print it as a percentage labelled `Accuracy (R² Score)`. The value is the R2 score, not a classification accuracy.
- `1_1linear_regression.ipynb`: `train_test_split(X, y, test_size=0.2, random_state=42)`.
- `multi_linear_reg.ipynb` trains on all 9 rows (no train/test split).

## 7. Advantages

- Simple, fast and easy to explain.
- Coefficients show the direction and size of each feature's effect.
- Has an exact solution; no tuning is needed for the basic model.
- Works well when the relationship really is close to linear.

## 8. Disadvantages

- Cannot capture curved relationships unless features are transformed (see `../Polynomial_Regression`).
- Sensitive to outliers, because errors are squared.
- Correlated features (multicollinearity) make individual coefficients unstable.
- Predictions outside the range of the training data (extrapolation) can be unreliable.
- Assumes errors with roughly constant variance and no strong dependence between observations.

## 9. Real-World Applications

- Estimating house or property prices from size, age and rooms.
- Estimating delivery or travel time from distance.
- Forecasting sales or demand from spend and seasonality features.
- Estimating vehicle fuel use or emissions from engine characteristics.
- Trend analysis in finance, economics and operations.

## 10. Python Libraries Used

In this table, Delivery Time is `Linear Regression_Delivery Time Prediction.ipynb` and House Price is `Multiple Linear Regression__House Price Prediction.ipynb`.

| Library | Used for | Notebooks |
|---|---|---|
| `pandas` | Loading CSVs, building DataFrames (also the one-row input DataFrame for a prediction) | all |
| `numpy` | `np.sqrt` for RMSE | `1_1linear_regression` |
| `matplotlib.pyplot` | Scatter plot with the fitted line (Delivery Time); actual vs predicted scatter plot (House Price, `1_1linear_regression`); residual histogram (`1_1linear_regression`) | Delivery Time, House Price, `1_1linear_regression` |
| `sklearn.linear_model` (`LinearRegression`, `linear_model`) | The model | all |
| `sklearn.model_selection.train_test_split` | Hold-out split | Delivery Time, House Price, `1_1linear_regression` |
| `sklearn.metrics` | `r2_score` (Delivery Time, House Price); `mean_absolute_error`, `mean_squared_error`, `r2_score` (`1_1linear_regression`) | Delivery Time, House Price, `1_1linear_regression` |
| `joblib` | Saving the fitted model to `delivery_model.pkl` and loading it again | Delivery Time |

## 11. Dataset Information

| Notebook | Data | Size | Notes |
|---|---|---|---|
| `Linear Regression_Delivery Time Prediction.ipynb` | `01_linear_delivery_time.csv` | 250 rows x 2 columns | Feature `Distance_km` (1.15 to 29.62); target `Delivery_Time_min` (9.6 to 119.1). No missing values. Practice dataset supplied with the author's regression practice questions. |
| `Multiple Linear Regression__House Price Prediction.ipynb` | `02_multiple_linear_house_price.csv` | 250 rows x 5 columns | Features `Area_sqft` (611 to 2,976), `Bedrooms` (1 to 5), `House_Age_years` (0 to 30), `Distance_to_CityCenter_km` (1.02 to 24.81); target `Price_Lakhs` (21.77 to 272.14). No missing values. Practice dataset supplied with the author's regression practice questions. |
| `1_1linear_regression.ipynb` | `homeprices.csv` | 50,000 rows x 4 columns | Features `area` (800 to 5,000), `bedrooms` (1 to 7), `age` (0 to 50); target `price` (217,000 to 967,000). No missing values. Appears to be synthetic data; origin not documented. |
| `multi_linear_reg.ipynb` | Inline Python dictionary | 9 rows x 4 columns | `ENGINESIZE`, `CYLINDERS`, `FUELCONSUMPTION_COMB` (features) and `CO2EMISSIONS` (target). A code comment says the values were typed in from an image, leaving out a row with a missing CO2 value. |

## 12. How to Execute

1. Set up the environment as described in the [root README](../../README.md#getting-started).
2. From the repository root run `python -m jupyter notebook`.
3. Open `01_Supervised_Learning/Linear_Regression/` and then a notebook. Jupyter runs the notebook from its own folder, so the relative CSV paths resolve.
4. Run the cells from top to bottom (Shift+Enter, or Kernel > Restart & Run All).

Three notebooks ask for values in a text box under the cell. Press Enter after each one.

**`Linear Regression_Delivery Time Prediction.ipynb`** asks for a distance twice: once with the model just trained, and once with the model re-loaded from `delivery_model.pkl`. Both use `float(...)`, so a non-numeric entry raises a `ValueError`.

| Prompt | What to type | Range in the data |
|---|---|---|
| `Enter the Km coverd:` (spelled this way in the code) | a number, e.g. `12.5` | 1.15 to 29.62 km |
| `Enter the Km covered:` | a number, e.g. `12.5` | 1.15 to 29.62 km |

Run the cells in order: the cell that loads `delivery_model.pkl` needs the file written by the cell before it. The file is written into this folder and is git-ignored (`*.pkl` in `.gitignore`).

**`Multiple Linear Regression__House Price Prediction.ipynb`** asks for four values in one cell. Bedrooms and house age use `int(...)`, so a decimal or non-numeric entry there raises a `ValueError`; the other two use `float(...)`.

| Prompt | What to type | Range in the data |
|---|---|---|
| `Enter the Sqft:` | a number, e.g. `1500` | 611 to 2,976 |
| `Enter No of Bedrooms:` | a whole number, e.g. `3` | 1 to 5 |
| `Enter the Age of the House:` | a whole number, e.g. `10` | 0 to 30 |
| `Enter the Distance from city:` | a number in km, e.g. `8.5` | 1.02 to 24.81 |

**`multi_linear_reg.ipynb`** asks for three values:

| Prompt | What to type | Range in the inline data |
|---|---|---|
| `Enter Engine Size (L):` | a decimal number, e.g. `2.4` | 1.5 to 3.7 |
| `Enter Number of Cylinders:` | a whole number, e.g. `4` | 4 or 6 |
| `Enter Fuel Consumption (L/100km):` | a decimal number, e.g. `9.6` | 5.9 to 11.6 |

If you type something that is not a number, the cell prints `Invalid input. Please enter numeric values.`

## 13. Expected Output

### `Linear Regression_Delivery Time Prediction.ipynb`

- The 250-row DataFrame, then the first rows of `X_train` and `X_test`, and `LinearRegression()` after fitting.
- The R2 score on the 50 test rows, printed as `Accuracy (R² Score): <value>%`.
- After the first distance input: `Time Taken to cover the <km> Km [<minutes>]`, followed by a scikit-learn `UserWarning` about missing feature names (a plain list is passed to `predict`).
- A scatter plot of all 250 rows (blue) with the fitted regression line (red), titled "Delivery Time vs Distance".
- `['delivery_model.pkl']` after saving and `Model Loaded Successfully LinearRegression()` after loading.
- After the second distance input: `Time Taken to cover <km> Km: <minutes> minutes`. This cell passes a DataFrame with the column name, so no warning appears.
- Saved output from the author's run:

| Cell | Saved output |
|---|---|
| R2 on the test set | `Accuracy (R² Score): 94.48%` |
| First prediction (input 34 km) | `Time Taken to cover the 34.0 Km [122.16329984]` |
| Second prediction, re-loaded model (input 34 km) | `Time Taken to cover 34.0 Km: 122.16329983675118 minutes` |

- 34 km is beyond the largest distance in the data (29.62 km), so the saved predictions are extrapolations. Because the split has no `random_state`, a new run gives a different R2 and different predictions.

### `Multiple Linear Regression__House Price Prediction.ipynb`

- The 250-row DataFrame (5 columns; pandas wraps the wide table in plain text).
- The second cell has a saved output, the `Price_Lakhs` column (`Length: 250`), although its current code only assigns `x` and `y` and displays nothing. The output comes from an earlier version of the cell.
- The first rows of `X_train` and `Y_test`, and `LinearRegression()` after fitting.
- The R2 score on the 50 test rows, printed as `Accuracy (R² Score): <value>%`.
- After the four inputs: `Predicted House Price: ₹<value> Lakhs`.
- A scatter plot of actual against predicted prices for the test rows (blue points), with a red dashed `Ideal Fit (Y = X)` diagonal.
- Saved output from the author's run:

| Cell | Saved output |
|---|---|
| R2 on the test set | `Accuracy (R² Score): 96.26%` |
| Prediction after the four inputs | `Predicted House Price: ₹229.33 Lakhs` (the values typed were not recorded) |

- Because the split has no `random_state`, a new run gives a different R2 and a different prediction.

### `1_1linear_regression.ipynb`

- Prints the shape `(50000, 4)`, a missing-value count per column (all 0) and `Train size: 40000 | Test size: 10000`.
- Prints MAE, MSE, RMSE and R2 on the test set.
- Shows two plots side by side: actual vs predicted price (with a red dashed diagonal), and a 50-bin histogram of residuals.
- Predicts a price for a new house with area 3000, 2 bedrooms, age 40.
- Saved output from the author's run:

| Metric (test set) | Saved output |
|---|---|
| MAE | 20,074.53 |
| MSE | 633,903,560.84 |
| RMSE | 25,177.44 |
| R2 | 0.9750 |
| Predicted price (3000, 2, 40) | 550,059.8 |

### `multi_linear_reg.ipynb`

- Shows the 9-row inline DataFrame, fits the model, then asks for the three inputs and prints them with `Predicted CO₂ Emission: <value> g/km`.
- Saved output from the author's run: `Engine Size: 2.4` and `Predicted CO₂ Emission: 198.37 g/km`. The other two input values were not recorded. The current code also prints the cylinder and fuel-consumption values, which the saved output does not include, so it comes from an earlier version of the cell.

## Changes (22 Nov 2026)

- **Notebook replacement.** `1_linear_regression.ipynb`, `2_linear_regression_multivariate.ipynb` and `houseprice.csv` were removed from the repository; they remain in the author's local folders. The two notebooks closely followed the codebasics "py" tutorials (code similarity 0.57 and 0.81), and `houseprice.csv` was the tutorial's 6-row house-price data. (`2_linear_regression_multivariate.ipynb` had received a `reg` to `model` fix and a re-run earlier the same day.)
- **Added** `Linear Regression_Delivery Time Prediction.ipynb` with `01_linear_delivery_time.csv`, and `Multiple Linear Regression__House Price Prediction.ipynb` with `02_multiple_linear_house_price.csv`: the author's own implementations (see [Source](#source)). Their saved outputs are the author's original runs and were not regenerated. In the Delivery Time notebook, the Windows username in the file path of the saved warning output was replaced with `<user>`.
- **Tests** (Python 3.10.11, scikit-learn 1.7.2, each notebook run from top to bottom on a copy): Delivery Time, 12 of 12 code cells ran (inputs 34 and 34); House Price, 9 of 9 code cells ran (inputs 2670, 1, 6, 6.89).
- `multi_linear_reg.ipynb`: the Windows username in file paths inside saved warning outputs was replaced with `<user>` (for example `C:\Users\<user>\AppData\...`). The warning text is otherwise unchanged. The same had been done in the two removed notebooks.
- This README: the Source and attribution note about the removed notebooks was replaced by the [Source](#source) note, and the files table and sections 6, 10, 11, 12 and 13 were rewritten for the current notebooks.
