# Importing necessary libraries
from keras.activations import softmax
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from sklearn.model_selection import learning_curve
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.layers import Dense
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.layers import Dropout
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import joblib
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.metrics import brier_score_loss

#Data retrieval
dataDerivation = 'E:\\Ashuju\\xin38\\derivation_lasso_nm.csv'
data_Derivation = pd.read_csv(dataDerivation)

X = data_Derivation.iloc[:, :-1]  # Features
Y = data_Derivation['label']

# Ensure X and Y are Numpy arrays
X = X.values if isinstance(X, pd.DataFrame) else X
Y = Y.values if isinstance(Y, pd.Series) else Y

# Set random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Learning rate scheduler function
def lr_scheduler(epoch, lr):
    if epoch % 100 == 0 and epoch > 0:
        lr = lr * 0.1
    return lr

# Hyperparameters and configurations
regularizer = l2(0.01)
optimizer = Adam(learning_rate=0.01)
activation_hidden = 'relu'
activation_output = 'sigmoid'
loss_function = 'binary_crossentropy'

# Create the MLP model
mlp = Sequential()
mlp.add(Dense(32, input_dim=X.shape[1], activation=activation_hidden, kernel_regularizer=regularizer))  # Input layer
mlp.add(Dense(16, activation=activation_hidden, kernel_regularizer=regularizer))  # Hidden layer
mlp.add(Dense(1, activation=activation_output))  # Output layer

# Compile the model
mlp.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

# Train the model using K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_idx, val_idx in kfold.split(X, Y):
    mlp.fit(X[train_idx], Y[train_idx], epochs=50, verbose=1,
            validation_data=(X[val_idx], Y[val_idx]),
            callbacks=[LearningRateScheduler(lr_scheduler)])

# Save the trained model
joblib.dump(mlp, 'mlp_model.pkl')

# Cross validation to obtain the AUC and its 95% confidence interval
X = data_Train.iloc[:, :-1]  # Features
Y = data_Train['label']
# Load the MLP model
model = joblib.load('mlp_model.pkl')

# Define cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=20)

# Record AUC for the training and testing sets
aucTrainList, aucTestList = [], []

# Cross-validation process
for train_index, test_index in rkf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Get predicted probabilities for the training and testing sets (directly use predicted probabilities)
    train_probs = model.predict_proba(
        X_train)  # If the model only returns one column of probabilities, [:, 1] is not needed
    test_probs = model.predict_proba(X_test)

    # If there is only one class probability, pass it directly to roc_auc_score
    aucTrainList.append(roc_auc_score(Y_train, train_probs))
    aucTestList.append(roc_auc_score(Y_test, test_probs))

# Calculate the mean and standard deviation of AUC for the training set
train_mean_auc = np.mean(aucTrainList)
train_std_auc = np.std(aucTrainList)

# Calculate the mean and standard deviation of AUC for the testing set
test_mean_auc = np.mean(aucTestList)
test_std_auc = np.std(aucTestList)

# Calculate 95% confidence intervals
confidence_level = 0.95
train_ci = stats.norm.interval(confidence_level, loc=train_mean_auc, scale=train_std_auc / np.sqrt(len(aucTrainList)))
test_ci = stats.norm.interval(confidence_level, loc=test_mean_auc, scale=test_std_auc / np.sqrt(len(aucTestList)))

# Output the mean, standard deviation, and 95% confidence intervals of AUC for training and testing sets
print('Training Set AUC:', round(train_mean_auc, 6))
print('Training Set AUC Standard Deviation:', round(train_std_auc, 6))
print('Training Set AUC 95% Confidence Interval:', (round(train_ci[0], 6), round(train_ci[1], 6)))

print('Testing Set AUC:', round(test_mean_auc, 6))
print('Testing Set AUC Standard Deviation:', round(test_std_auc, 6))
print('Testing Set AUC 95% Confidence Interval:', (round(test_ci[0], 6), round(test_ci[1], 6)))

# Cross-validation to obtain the Brier Score and its 95% confidence interval
# Load the MLP model
model = joblib.load('mlp_model.pkl')  # Load the pre-trained model

# Define cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=20)

# Lists to record Brier Scores
train_brier_scores, test_brier_scores = [], []

# Cross-validation evaluation
for train_index, test_index in rkf.split(X):
    # Split the data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Use the pre-trained model to make predictions (no training here)
    train_probs = model.predict_proba(X_train)  # Directly use predict_proba output
    test_probs = model.predict_proba(X_test)

    # Compute Brier Score for training and testing sets
    train_brier_score = brier_score_loss(Y_train, train_probs)
    test_brier_score = brier_score_loss(Y_test, test_probs)

    train_brier_scores.append(train_brier_score)
    test_brier_scores.append(test_brier_score)

# Calculate the mean and standard deviation of Brier Scores for training and testing sets
mean_train_brier = np.mean(train_brier_scores)
std_train_brier = np.std(train_brier_scores)

mean_test_brier = np.mean(test_brier_scores)
std_test_brier = np.std(test_brier_scores)

# Compute 95% confidence intervals
confidence_level = 0.95
train_brier_ci = stats.norm.interval(confidence_level, loc=mean_train_brier,
                                     scale=std_train_brier / np.sqrt(len(train_brier_scores)))
test_brier_ci = stats.norm.interval(confidence_level, loc=mean_test_brier,
                                    scale=std_test_brier / np.sqrt(len(test_brier_scores)))

# Output the Brier Score evaluation results
print('Training Set Brier Score Mean:', round(mean_train_brier, 6))
print('Training Set Brier Score Standard Deviation:', round(std_train_brier, 6))
print('Training Set Brier Score 95% Confidence Interval:', (round(train_brier_ci[0], 6), round(train_brier_ci[1], 6)))

print('Testing Set Brier Score Mean:', round(mean_test_brier, 6))
print('Testing Set Brier Score Standard Deviation:', round(std_test_brier, 6))
print('Testing Set Brier Score 95% Confidence Interval:', (round(test_brier_ci[0], 6), round(test_brier_ci[1], 6)))