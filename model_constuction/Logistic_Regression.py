# Import necessary libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedKFold

#Data retrieval
dataDerivation ='E:\\Ashuju\\xin38\\derivation_lasso_nm.csv'
data_Derivation = pd.read_csv(dataDerivation)

X = data_Derivation.iloc[:, :-1]
Y = data_Derivation['label']

# Logistic Regression parameter grid
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Grid search
lr_estimator = LogisticRegression(solver='liblinear', max_iter=1000)
grid_search_lr = GridSearchCV(estimator=lr_estimator, param_grid=param_grid_lr, cv=5)
grid_search_lr.fit(X, Y)

# Output the best parameters
best_params_lr = grid_search_lr.best_params_
print("Best parameters for Logistic Regression:", best_params_lr)

# Model construction
model = LogisticRegression(C=best_params_lr['C'], penalty=best_params_lr['penalty'], solver='liblinear', max_iter=1000)
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

#Cross validation to obtain the Brier Scores and its 95% confidence interval
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