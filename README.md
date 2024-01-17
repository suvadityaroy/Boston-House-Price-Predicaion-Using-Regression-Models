# Boston Housing Price Prediction

## Index
1. [Introduction](#introduction)
2. [About the Dataset](#about-the-dataset)
3. [Project Explanation](#project-explanation)
   1. [Data Collection](#data-collection)
   2. [Loading the Collected Data](#loading-the-collected-data)
   3. [Feature Engineering](#feature-engineering)
   4. [Feature Selection](#feature-selection)
   5. [Building Machine Learning Models](#building-machine-learning-models)
   6. [Model Accuracy of Different Models](#model-accuracy-of-different-models)
   7. [Model Performances](#model-performances)
   8. [Building Optimum Model](#building-optimum-model)
4. [Conclusion](#conclusion)

## Introduction

This project focuses on predicting house prices in Boston, Massachusetts, using machine learning algorithms. The dataset utilized in this project was collected by the U.S Census Service and includes various features related to housing. The dataset has been widely used in the literature to benchmark algorithms.

## About the Dataset

- **Origin:** Natural
- **Usage:** Assessment
- **Number of Cases:** 506
- **Variables:**
  1. CRIM - Per capita crime rate by town
  2. ZN - Proportion of residential land zoned for lots over 25,000 sq.ft.
  3. INDUS - Proportion of non-retail business acres per town.
  4. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
  5. NOX - Nitric oxides concentration (parts per 10 million)
  6. RM - Average number of rooms per dwelling
  7. AGE - Proportion of owner-occupied units built prior to 1940
  8. DIS - Weighted distances to five Boston employment centers
  9. RAD - Index of accessibility to radial highways
  10. TAX - Full-value property-tax rate per 10,000 dollars.
  11. PTRATIO - Pupil-teacher ratio by town
  12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  13. LSTAT - % lower status of the population
  14. MEDV - Median value of owner-occupied homes in 1000's dollars (prediction target)

## Project Explanation

### Data Collection

The dataset was obtained from the StatLib archive and Kaggle. It consists of 506 cases and 14 features.

### Loading the Collected Data

The dataset is loaded using the `read_csv` method in the pandas library.

```python
# Initializing column names
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Loading Boston Housing Dataset
boston = pd.read_csv('../data/housing.csv', delimiter=r"\s+", names=columns)
```
### Feature Engineering

Null values and data types of each feature are checked. Fortunately, there are no null values in any of the features.

```python
# Check for null values
boston.isnull().sum()
```

```python
# Check for data types of all the columns
boston.dtypes
```

### Feature Selection

Correlation analysis and ExtraTreesRegressor are used to identify the most important features.

```python
# Correlation with MEDV
corr_with_medv[:-1].abs().sort_values(ascending=False)
```

```python
# Feature Impotances by ExtraTressRegressor
important_features.sort_values(ascending=False)
```

### Building Machine Learning Models

Linear Regression, Decision Tree Regression, and Random Forest Regression models are implemented.

```python
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
```

```python
# Decision Tree Regression
dtr_model = DecisionTreeRegressor(max_depth=23, random_state=3)
dtr_model.fit(X_train[:, :], y_train)
```

```python
# Random Forest Regression
rfr = RandomForestRegressor(max_depth=7, random_state=63)
rfr.fit(X_train, y_train)
```

### Model Accuracy of Different Models

Model accuracy is evaluated for Linear Regression, Decision Tree Regression, Random Forest Regression, and k Neighbors Regression.

### Model Performances

Performance metrics such as score and mean squared error are computed for each model.

### Building Optimum Model

An optimal Random Forest Regressor model is built based on the chosen parameters for random state and max depth.

```python
# Building Optimal Random Forest Regressor Model
random_forest_regressor = RandomForestRegressor(max_depth=13, random_state=68)
random_forest_regressor.fit(X_train, y_train)
```

## Conclusion

The project concludes with the development of a Random Forest Regressor Model that achieves a high training accuracy of 97.89% and testing accuracy of 96.73%. This model is built using the top 6 features identified through feature selection techniques.
```

Feel free to further customize it as needed!
