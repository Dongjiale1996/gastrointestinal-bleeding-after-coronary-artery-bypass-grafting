# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.ensemble import RandomForestClassifier as RFC  # For Random Forest classifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold  # For model evaluation and hyperparameter tuning
from sklearn.preprocessing import StandardScaler  # For data normalization
from sklearn.feature_selection import RFECV, RFE  # For feature selection
from sklearn.metrics import accuracy_score  # For model evaluation metrics
import matplotlib.pyplot as plt  # For data visualization

#Load the dataset
data_file = 'E:\\Ashuju\\xin43\\derivation_nm.csv'  # Path to the dataset
data = pd.read_csv(data_file)  # Read the dataset using pandas
X = data.iloc[:, 0:-1]  # Extract features (all columns except the last one)
y = data['label']  # Extract target labels
print(X.shape)  # Print the shape of the feature dataset

# Set up the parameter grid for Random Forest hyperparameter tuning
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used
}

# Initialize the Random Forest classifier
rf = RFC(random_state=42)

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=4, verbose=2)  # n_jobs=4 for parallel processing

grid_search.fit(X, y)  # Fit the model with grid search

# Print the best parameters from the grid search
print("Best parameters from grid search:")
print(grid_search.best_params_)  # Print the best parameters


best_rf = grid_search.best_estimator_  # Get the best estimator from grid search
# Recursive Feature Elimination with Cross-Validation (RFECV)
rfecv = RFECV(
    estimator=best_rf,  # Use the best Random Forest classifier
    step=1,  # Number of features to remove at each iteration
    cv=StratifiedKFold(5),  # 5-fold cross-validation
    scoring='f1_weighted',  # Scoring metric
    n_jobs=-1,  # Use all available CPUs
    verbose=2  # Verbose output
)
rfecv.fit(X, y)  # Fit RFECV with the dataset

# Print the number of optimal features
print("Optimal number of features: %d" % rfecv.n_features_)

# Output the selected features
selected_features = [f for f, s in zip(X.columns, rfecv.support_) if s]
print("Selected features:")
print(selected_features)

# Calculate the mean cross-validation score
mean_cv_score = rfecv.cv_results_['mean_test_score'].mean()  # Average cross-validation score
print("Mean cross-validation score: %.2f" % mean_cv_score)
