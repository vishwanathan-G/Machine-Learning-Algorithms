# Big Data with PySpark

## Files in this folder

| File | Description |
|---|---|
| `RDD.ipynb` | Spark Core basics: `SparkContext`, `parallelize`, `map`, `filter`, `collect`, `count`, `first`, `take`, followed by a revision section with one example per function |
| `Dataframe.ipynb` | Spark DataFrame basics: `SparkSession`, `createDataFrame`, `show`, `select`, `filter`, `col`, `withColumn`, `drop`, followed by a revision section |
| `sql.ipynb` | Spark SQL: registering a DataFrame as a temporary view and querying it with `SELECT`, `WHERE`, `GROUP BY`, `AVG`, `COUNT` and `AS`, followed by a revision section |
| `PySparkML.ipynb` | Spark MLlib: `StringIndexer`, `VectorAssembler`, a logistic regression classifier on toy data, and a linear regression model (salary from years of experience) with `randomSplit` and `RegressionEvaluator` (R2), followed by revision notes |
| `pyspark_salary_dataset_1000.csv` | Salary data read by `PySparkML.ipynb`: 1,000 rows x 3 columns (`ID`, `Experience`, `Salary`) |

All four notebooks were written and run in Google Colab. Each one first runs the examples with saved outputs, then has a revision section that repeats each function with a definition, syntax and a short example. The expected output for each revision example is written in a markdown cell; the revision code cells themselves have no saved outputs.

## 1. Algorithm Name

Apache Spark with PySpark: distributed data processing (RDD, DataFrame, Spark SQL) and distributed machine learning with MLlib (Linear Regression, Logistic Regression).

## 2. Definition

Apache Spark is an open-source engine for processing large datasets in parallel across a cluster of machines, or across the cores of one machine in local mode. PySpark is its Python API.

- **RDD (Resilient Distributed Dataset)**: Spark's low-level abstraction, an immutable collection of elements split into partitions that are processed in parallel.
- **DataFrame**: a distributed table with named columns and a schema. Its queries are optimised by Spark's Catalyst optimiser.
- **Spark SQL**: lets you query DataFrames with SQL after registering them as views.
- **MLlib** (`pyspark.ml`): Spark's machine learning library, with feature transformers, estimators (models) and evaluators that run on DataFrames.

## 3. How It Works

1. A **driver** program creates a `SparkContext` (RDD API) or a `SparkSession` (DataFrame, SQL and ML APIs).
2. Data is split into **partitions**, which executors process in parallel. In local mode the executors are threads on your machine.
3. **Transformations** (`map`, `filter`, `select`, `withColumn`) are lazy. They only add steps to an execution plan (a DAG).
4. **Actions** (`collect`, `count`, `first`, `take`, `show`) trigger the computation and return results to the driver.
5. In MLlib, transformers such as `VectorAssembler` add columns. An estimator such as `LinearRegression` is trained with `.fit(df)`, which returns a model, and `model.transform(df)` adds a `prediction` column. Evaluators then score those predictions.

## 4. Why It Is Used

- To process data that is too large for one machine's memory, or too slow to process on one core.
- The same code runs on a laptop (local mode) and on a cluster.
- One engine covers ETL, SQL analytics, streaming and machine learning.
- DataFrames and SQL make big-data work accessible to people who know pandas or SQL.

## 5. Mathematical Intuition

The MLlib models in `PySparkML.ipynb` are the same models as their scikit-learn equivalents, trained in a distributed way:

- **Linear Regression**: `y_hat = b0 + b1*x`, fitted by minimising squared error, `(1/n) * sum((y - y_hat)^2)`. Spark adds optional L1/L2 regularisation (`regParam`, `elasticNetParam`). See [`../01_Supervised_Learning/Linear_Regression`](../01_Supervised_Learning/Linear_Regression/README.md).
- **Logistic Regression**: `p = 1 / (1 + e^(-z))` with `z = b0 + b1*x1 + b2*x2`; predicts 1 when `p >= threshold` (0.5 by default). See [`../01_Supervised_Learning/Logistic_Regression`](../01_Supervised_Learning/Logistic_Regression/README.md).
- **R2** (`RegressionEvaluator(metricName="r2")`): `R2 = 1 - SS_res / SS_tot`.
- `randomSplit([0.8, 0.2], seed=42)` assigns each row to a split at random with those probabilities, so the split sizes are close to 80/20 but not exact.
- `StringIndexer` maps categories to indices ordered by frequency (the most frequent gets 0.0).

## 6. Important Parameters

**General**

| API | Parameter | Default | Meaning |
|---|---|---|---|
| `SparkContext(master, appName)` | `master` | (required here) | `"local"` runs on one thread; `"local[*]"` uses all cores; a cluster URL runs on a cluster |
| `SparkSession.builder.appName(...).getOrCreate()` | `appName` | (none) | Name shown in the Spark UI; `getOrCreate` reuses an existing session |
| `sc.parallelize(data, numSlices)` | `numSlices` | Spark default | Number of partitions |
| `spark.read.csv(path, header, inferSchema)` | `header`, `inferSchema` | `False`, `False` | Use the first row as column names; detect column types |
| `StringIndexer` | `inputCol`, `outputCol`, `handleInvalid` | `handleInvalid='error'` | Category column to index |
| `VectorAssembler` | `inputCols`, `outputCol` | (required) | Combine columns into one `features` vector |
| `LinearRegression` | `featuresCol`, `labelCol`, `maxIter`, `regParam`, `elasticNetParam` | `'features'`, `'label'`, `100`, `0.0`, `0.0` | Columns and regularisation |
| `LogisticRegression` | `featuresCol`, `labelCol`, `maxIter`, `regParam`, `threshold` | `'features'`, `'label'`, `100`, `0.0`, `0.5` | Columns, regularisation, decision threshold |
| `randomSplit(weights, seed)` | `weights`, `seed` | (required), random | Split proportions and seed |
| `RegressionEvaluator` | `metricName` | `'rmse'` | `'rmse'`, `'mse'`, `'mae'`, `'r2'` |

**Used in these notebooks**

- `RDD.ipynb`: `SparkContext("local", "RDDdemo")`.
- `Dataframe.ipynb`, `sql.ipynb`: `SparkSession.builder.appName("test").getOrCreate()`. The revision sections use other app names.
- `PySparkML.ipynb`:
  - `StringIndexer(inputCol="Gender", outputCol="GenderIndex")`
  - `VectorAssembler(inputCols=["Age", "Salary"], outputCol="features")` for the toy examples
  - `LogisticRegression(featuresCol="features", labelCol="label")`
  - `spark.read.csv("/content/pyspark_salary_dataset_1000.csv", header=True, inferSchema=True)` (the Colab path; see section 12 for running locally)
  - `VectorAssembler(inputCols=["Experience"], outputCol="features")`, with `Salary` renamed to `label`
  - `randomSplit([0.8, 0.2], seed=42)`
  - `LinearRegression(featuresCol="features", labelCol="label")` with default `maxIter`, no regularisation
  - `RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")`

## 7. Advantages

- Scales from one laptop to large clusters without code changes.
- In-memory processing is much faster than disk-based MapReduce for iterative work such as machine learning.
- Lazy evaluation lets Spark optimise the whole pipeline before running it.
- One engine for SQL, DataFrames, streaming and ML.
- Fault tolerance: lost partitions are recomputed from their lineage.

## 8. Disadvantages

- High overhead for small data; pandas and scikit-learn are faster on a few thousand rows.
- Needs a Java runtime and more setup than plain Python libraries.
- Debugging distributed, lazily evaluated code is harder.
- MLlib offers fewer algorithms and options than scikit-learn.
- `collect()` on large data can exhaust the driver's memory.

## 9. Real-World Applications

- ETL pipelines that clean and join large log, transaction or event datasets.
- SQL analytics on data lakes.
- Training models on very large tables, such as click-through prediction or churn prediction.
- Recommendation systems (MLlib's ALS) and large-scale feature engineering.
- Near-real-time processing with Structured Streaming.

## 10. Python Libraries Used

| Library | Used for | Notebooks |
|---|---|---|
| `pyspark.SparkContext` | RDD entry point | `RDD` |
| `pyspark.sql.SparkSession` | DataFrame, SQL and ML entry point | `Dataframe`, `sql`, `PySparkML` |
| `pyspark.sql.functions.col` | Column expressions | `Dataframe` |
| `pyspark.ml.feature` (`StringIndexer`, `VectorAssembler`) | Feature engineering | `PySparkML` |
| `pyspark.ml.classification.LogisticRegression` | Classification | `PySparkML` |
| `pyspark.ml.regression.LinearRegression` | Regression | `PySparkML` |
| `pyspark.ml.evaluation.RegressionEvaluator` | R2 score | `PySparkML` |
| `google.colab.files` | Uploading the CSV in Google Colab (Colab only) | `PySparkML` |

## 11. Dataset Information

| Notebook | Data | Notes |
|---|---|---|
| `RDD.ipynb` | Inline Python lists | `[11, 22, 33, 44, 55]`, `[1, 2, 3, 4]`, `[1, 2, 3, 4, 5, 6]` |
| `Dataframe.ipynb` | Inline list of tuples | 10 employee rows with columns `id`, `name`, `salary` (10,000 to 100,000); first names used as sample values. The revision section uses a 2-row subset. |
| `sql.ipynb` | Inline list of tuples | The same 10 employees with an extra `Dept` column (`IT`, `HR`, `Finance`, `Admin`, `R&D`), registered as the view `emp`. The revision section adds small 2-row examples. |
| `PySparkML.ipynb` (feature and classification examples) | Inline tuples | 3 rows: `Gender`/`Age`; `Age`/`Salary`; and `Age`/`Salary`/`label` (0 or 1) |
| `PySparkML.ipynb` (linear regression) | `pyspark_salary_dataset_1000.csv` (included in this folder) | 1,000 rows with a header row: `ID` (integer), `Experience` (years, decimal, e.g. 6.1) and `Salary` (integer, e.g. 48760). Only `Experience` (feature) and `Salary` (label) are used. In Colab the file was uploaded and read from `/content/pyspark_salary_dataset_1000.csv`. The file was recovered from the author's own files, and its rows match the notebook's saved output. |

## 12. How to Execute

**Option A: Google Colab (how the notebooks were written)**

1. Upload a notebook to Google Colab.
2. If `import pyspark` fails, run `!pip install pyspark` in a new cell.
3. Run the cells from top to bottom. In `PySparkML.ipynb`, the `files.upload()` cell opens a file picker. Choose `pyspark_salary_dataset_1000.csv` from this folder.

**Option B: local Jupyter**

1. Set up the base environment as described in the [root README](../README.md#getting-started).
2. Install a Java runtime. `requirements.txt` notes Java 8, 11 or 17. Spark 4.x requires Java 17 or later. Set `JAVA_HOME` to the Java installation folder.
3. Install PySpark, which is optional in `requirements.txt`:
   ```
   python -m pip install "pyspark>=3.4"
   ```
4. Run `python -m jupyter notebook` from the repository root and open a notebook in `04_Big_Data_PySpark/`.
5. Run the cells in order. Notes for running locally:
   - In `PySparkML.ipynb`, skip the `from google.colab import files` cell (it works only in Colab). `pyspark_salary_dataset_1000.csv` is already next to the notebook; change the path in the `spark.read.csv(...)` cell from `/content/pyspark_salary_dataset_1000.csv` to `pyspark_salary_dataset_1000.csv`.
   - `RDD.ipynb` stops its `SparkContext` (`sc.stop()`) at the end of the first part and creates a new one at the start of the revision section, so run the cells in order.
   - On Windows, Spark may print warnings about `HADOOP_HOME` or `winutils.exe`. If the Python workers fail to start, set the `PYSPARK_PYTHON` environment variable to the Python executable of your virtual environment.

**Local test (22 Nov 2026).** `PySparkML.ipynb` was run on a copy with Python 3.13.7, PySpark 4.1.1 and Java 17, with the Colab upload cell skipped and the `/content/` path pointed at the local CSV. (PySpark 4.1.1 is newer than the `<4` pin above, which `requirements.txt` sets because PySpark 4.x needs Java 17 or newer.) The CSV-based part reproduced the saved Colab outputs exactly: the data preview, the `VectorAssembler` output, the `randomSplit` counts (838 training and 162 test rows), the `LinearRegression` predictions and `R2 Score: 0.9958205217455208`.

The cells that create DataFrames or RDDs from Python lists could not be run locally: `PySparkML.ipynb` cells 6 to 8 and 21 (cell numbers count from 0, markdown cells included), and the `RDD`, `Dataframe` and `sql` notebooks. On the test laptop Spark's Python worker crashed even for a three-line `parallelize().collect()` test, so this is a problem of that local environment. The saved outputs of those cells come from Google Colab, where they ran.

No notebook in this folder asks for keyboard input.

## 13. Expected Output

### `RDD.ipynb` (saved output from the author's run)

| Operation | Saved output |
|---|---|
| `sc.parallelize([11, 22, 33, 44, 55]).collect()` | `[11, 22, 33, 44, 55]` |
| `map(lambda x: x*x)` on `[1, 2, 3, 4]` | `[1, 4, 9, 16]` |
| `filter(lambda x: x % 2 == 0)` on `[1..6]` | `[2, 4, 6]` |
| `count()` | `6` |
| `first()` / `take(2)` | `11` / `[11, 22]` |
| `rdd.map(lambda x: x*2)` without an action | `PythonRDD[8] at RDD at PythonRDD.scala:57` (a lazy RDD reference) |
| `.collect()` of that RDD | `[22, 44, 66, 88, 110]` |

The revision section prints the same kinds of results, and its markdown cells state the expected values (for example `[11, 22, 33]`, `6`, `11`, `[11, 22]`, `SparkContext stopped.`).

### `Dataframe.ipynb` (saved output from the author's run)

- `df.show()`: the 10-row `id | name | salary` table.
- `select("name")`: the name column only.
- `filter(df.salary > 50000)`: the 5 rows with salary 60,000 to 100,000.
- `withColumn("Bonus", col("salary") + 10000)`: adds `Bonus` (20,000 to 110,000).
- `drop("salary")`: the table with `id`, `name`, `Bonus`.
- The revision section produces the 2-row tables shown in its markdown cells.

### `sql.ipynb` (saved output from the author's run)

| Query | Saved output |
|---|---|
| `select * from emp` | all 10 rows with `Dept` |
| `select * from emp where salary>50000` | 5 rows (salary 60,000 to 100,000) |
| `select dept, avg(salary) from emp group by dept` | HR 45,000; Finance 30,000; Admin 70,000; IT 40,000; R&D 85,000 |
| `select count(*) as tot_emp from emp` | 10 |

The revision section adds `SELECT name, Dept`, `WHERE Dept = 'IT'` (3 rows), `GROUP BY` with `COUNT(*)`, `AVG`, `COUNT` and `AS` aliases. Its markdown cells show the expected tables. The order of rows in `GROUP BY` results is not guaranteed.

### `PySparkML.ipynb` (saved output from the author's run)

The saved outputs are the author's Google Colab outputs. The last column shows the result of the local test described in section 12.

| Step | Saved output | Local test |
|---|---|---|
| `StringIndexer` on `Gender` (cell 6) | `Male` → 0.0, `Female` → 1.0 | Not run (DataFrame from a Python list) |
| `VectorAssembler` on `Age`, `Salary` (cell 7) | `features` column such as `[25.0,50000.0]` | Not run (DataFrame from a Python list) |
| Logistic regression on 3 toy rows (cell 8) | predictions 0.0, 1.0, 1.0, the same as the labels (trained and evaluated on the same 3 rows) | Not run (DataFrame from a Python list) |
| `df.show(10)` on the salary CSV | first 10 rows of `ID`, `Experience`, `Salary` | Reproduced exactly |
| `VectorAssembler` on `Experience` | `features` column such as `[6.1]` | Reproduced exactly |
| `randomSplit([0.8, 0.2], seed=42)` | `Training Data Count: 838`, `Testing Data Count: 162` | Reproduced exactly |
| `predictions.show()` | test rows with `features`, `label` and `prediction` | Reproduced exactly |
| `RegressionEvaluator(metricName="r2")` | `R2 Score: 0.9958205217455208` | Reproduced exactly |
| Prediction for `Experience = 12.0` (cell 21) | 80,075.38 | Not run (DataFrame from a Python list) |

Cell 9 is a markdown note, not code (see Changes below). With the included CSV these are the expected values; a different salary CSV would give different counts, R2 score and prediction.

## Changes (22 Nov 2026)

- `PySparkML.ipynb`: cell 9 contained `model = LogisticRegression()` followed by `model.fit(x,y)`. That is the scikit-learn style: `x` and `y` are never defined in the notebook, and Spark estimators are trained on a DataFrame. The cell is now a markdown note that explains this; the cell above it already trains with `lr.fit(final_data)`. The saved Colab outputs of the other cells are kept.
- `pyspark_salary_dataset_1000.csv`: added. The file (1,000 rows; columns `ID`, `Experience`, `Salary`) was recovered from the author's own files, and its rows match the notebook's saved output.
- `RDD.ipynb`, `Dataframe.ipynb`, `sql.ipynb`: unchanged.
- This README: the dataset is described as included, the local-run steps use it, the note about the old `model.fit(x,y)` cell was removed, and the local test results were added.
