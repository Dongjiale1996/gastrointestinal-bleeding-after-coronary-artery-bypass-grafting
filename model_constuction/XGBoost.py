# Importing necessary libraries
from sklearn.metrics import precision_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn, unpatch_sklearn
from scipy import stats
from sklearn.metrics import brier_score_loss

patch_sklearn()

# Data retrieval
dataDerivation ='E:\\Ashuju\\xin38\\derivation_lasso_nm.csv'
data_Derivation = pd.read_csv(dataDerivation)
# Define features (X) and labels (Y)
X = data_Derivation.iloc[:, :-1]
Y = data_Derivation['label']

# Define the parameter ranges for GridSearchCV
parameters = {
    'n_estimators': np.arange(8, 15, 1),
    'max_depth': np.arange(1, 6, 1),
    'learning_rate': np.arange(0.3, 0.5, 0.05),
    'gamma': np.arange(0, 0.03, 0.01),
    'min_child_weight': np.arange(0, 5, 1),
}

# Initialize the XGBoost classifier
clf = xgb.XGBClassifier(n_jobs=-4, random_state=90)

# Perform GridSearchCV
GS = GridSearchCV(clf, parameters, scoring='roc_auc', cv=5)
GS.fit(X, Y)

# Print the best parameters found by GridSearchCV
print("Best Parameters:", GS.best_params_)

n_estimators=GS.best_params_['n_estimators']
max_depth=GS.best_params_['max_depth']
learning_rate=GS.best_params_['learning_rate']
gamma=GS.best_params_['gamma']
min_child_weight=GS.best_params_['min_child_weight']

# Model construction
model=xgb.XGBClassifier(learning_rate =learning_rate,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        n_jobs=--4, random_state=90)
model.fit(X, Y)

#Cross validation to obtain the AUC and its 95% confidence interval
# Initialize the cross-validator
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=20)

# Lists to record scores and AUC values
aucTrainList, aucTestList = [], []

# Cross-validation evaluation
for train_index, test_index in rkf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Train the model
    model.fit(X_train, Y_train)

    # Predict for training and test sets
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Compute AUC for training and test sets
    aucTrainList.append(roc_auc_score(Y_train, y_train_pred))
    aucTestList.append(roc_auc_score(Y_test, y_test_pred))

# Calculate the mean and standard deviation of AUC
train_mean_auc = np.mean(aucTrainList)
train_std_auc = np.std(aucTrainList)
test_mean_auc = np.mean(aucTestList)
test_std_auc = np.std(aucTestList)

# Calculate 95% confidence intervals
confidence_level = 0.95
train_ci = stats.norm.interval(confidence_level, loc=train_mean_auc, scale=train_std_auc / np.sqrt(5))
test_ci = stats.norm.interval(confidence_level, loc=test_mean_auc, scale=test_std_auc / np.sqrt(5))

# Output model evaluation results
print('Model training set average AUC:', round(train_mean_auc, 6))
print('Training set AUC standard deviation:', round(train_std_auc, 6))
print('Training set AUC 95% confidence interval:', (round(train_ci[0], 6), round(train_ci[1], 6)))

print('Model test set average AUC:', round(test_mean_auc, 6))
print('Test set AUC standard deviation:', round(test_std_auc, 6))
print('Test set AUC 95% confidence interval:', (round(test_ci[0], 6), round(test_ci[1], 6)))

#Cross validation to obtain the Brier Score and its 95% confidence interval
# Initialize the cross-validator
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=20)

# Lists to record Brier Scores
train_brier_scores = []
test_brier_scores = []

# Cross-validation evaluation
for train_index, test_index in rkf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Train the model
    model.fit(X_train, Y_train)

    # Predict probabilities for training and test sets
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]

    # Calculate Brier Score
    train_brier_score = brier_score_loss(Y_train, y_train_pred)
    test_brier_score = brier_score_loss(Y_test, y_test_pred)

    train_brier_scores.append(train_brier_score)
    test_brier_scores.append(test_brier_score)

# Calculate mean and standard deviation of Brier Scores for training and test sets
mean_train_brier = np.mean(train_brier_scores)
std_train_brier = np.std(train_brier_scores)

mean_test_brier = np.mean(test_brier_scores)
std_test_brier = np.std(test_brier_scores)

# Calculate 95% confidence intervals
confidence_level = 0.95
train_ci = stats.norm.interval(confidence_level, loc=mean_train_brier, scale=std_train_brier / np.sqrt(len(train_brier_scores)))
test_ci = stats.norm.interval(confidence_level, loc=mean_test_brier, scale=std_test_brier / np.sqrt(len(test_brier_scores)))

# Output Brier Score evaluation results
print('Training set Brier Score mean:', round(mean_train_brier, 6))
print('Training set Brier Score standard deviation:', round(std_train_brier, 6))
print('Training set Brier Score 95% confidence interval:', (round(train_ci[0], 6), round(train_ci[1], 6)))

print('Test set Brier Score mean:', round(mean_test_brier, 6))
print('Test set Brier Score standard deviation:', round(std_test_brier, 6))
print('Test set Brier Score 95% confidence interval:', (round(test_ci[0], 6), round(test_ci[1], 6)))