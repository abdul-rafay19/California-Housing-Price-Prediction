# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn tools
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# For randomized search
from scipy.stats import randint

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Initial data check
print(df.head())
print(df.info())
print(df.describe())

# Histogram for distribution of numerical features
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Correlation matrix heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot to visualize relation between income and house value
df.plot(kind="scatter", x="MedInc", y="MedHouseVal", alpha=0.1)
plt.title("Median Income vs. Median House Value")
plt.show()

# Splitting the data into training and test sets
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
housing_train = train_set.copy()
housing_test = test_set.copy()

# Separating features and labels
housing = housing_train.drop("MedHouseVal", axis=1)
housing_labels = housing_train["MedHouseVal"].copy()

# List of numerical features (all in this dataset)
num_attribs = housing.select_dtypes(include=[np.number]).columns.tolist()

# Pipeline for numerical data preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Full preprocessing pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
])

# Prepare the data
housing_prepared = full_pipeline.fit_transform(housing)

# Train models
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# Function to display cross-validation RMSE scores
def display_scores(model, features, labels):
    scores = cross_val_score(model, features, labels,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    print(f"Model: {model.__class__.__name__}")
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("Standard deviation:", rmse_scores.std())
    print("")

# Evaluate models
display_scores(tree_reg, housing_prepared, housing_labels)
display_scores(forest_reg, housing_prepared, housing_labels)

# Grid Search for best hyperparameters for Random Forest
param_grid = [
    {'n_estimators': [30, 50, 100], 'max_features': [4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [30, 50], 'max_features': [4, 6]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print("Best parameters from Grid Search:")
print(grid_search.best_params_)

# Randomized Search (faster alternative to grid search)
param_dist = {
    'n_estimators': randint(30, 100),
    'max_features': randint(2, 8),
}

random_search = RandomizedSearchCV(forest_reg, param_distributions=param_dist,
                                   n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                   random_state=42, return_train_score=True)
random_search.fit(housing_prepared, housing_labels)

print("Best parameters from Randomized Search:")
print(random_search.best_params_)

# Evaluate final model from RandomizedSearchCV on test set
final_model = random_search.best_estimator_

# Prepare the test set
X_test = housing_test.drop("MedHouseVal", axis=1)
y_test = housing_test["MedHouseVal"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("\nFinal Model Evaluation on Test Set:")
print("RMSE:", final_rmse)
