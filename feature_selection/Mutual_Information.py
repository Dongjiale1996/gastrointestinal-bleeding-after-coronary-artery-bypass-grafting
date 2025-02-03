# Import necessary libraries
from sklearn.feature_selection import mutual_info_classif as MI
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt

# Load dataset
dataFile = 'E:\\Ashuju\\xin43\\derivation_nm.csv'
data = pd.read_csv(dataFile)

# Split data into features (X) and target variable (Y)
X = data.iloc[:, 0:-1]  # Features
Y = data['label']       # Target variable

# Calculate Mutual Information (MI) scores using repeated 5-fold cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=20)
MI_scores = []
for train_index, test_index in rkf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    MI_result = MI(X_train, Y_train)  # Compute MI scores for each split
    MI_scores.append(MI_result)

# Aggregate MI scores across folds and calculate the mean
MI_scores_array = np.array(MI_scores)
MI_score_mean = MI_scores_array.mean(axis=0)

# Sort scores and prepare feature labels
sorted_scores = np.sort(MI_score_mean)  # Sort MI scores in ascending order
sorted_indices = np.argsort(MI_score_mean)  # Get indices for sorted scores

features = X.columns[sorted_indices]
# Plot feature score distribution to identify inflection points
plt.figure(figsize=(10, 6))
plt.plot(features, sorted_scores, marker='o', linestyle='-', color='b')  # Line chart
plt.xticks(rotation=90)
plt.xlabel("Features (sorted by score)")
plt.ylabel("MI Score (Mean)")
plt.title("Mean MI Score of Features")
plt.tight_layout()
plt.show()

# Create a DataFrame to store feature names and their mean MI scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'MI Score Mean': MI_score_mean
})
sorted_features = feature_scores.sort_values(by='MI Score Mean', ascending=False).reset_index(drop=True)
sorted_features.index += 1  # Adjust index to start from 1
# Display all features and their MI scores
print("All features and their MI scores:")
print(sorted_features)

# Select the score threshold based on the feature score distribution chart.
# In this study, features with a retention MI Score mean greater than 0.005 were selected for further analysis.
filtered_features = sorted_features[sorted_features['MI Score Mean'] > 0.005]

# Save the filtered features and their scores to a CSV file
csv_path = 'E:\\Ashuju\\xin43\\Features_MI.csv'
filtered_features.to_csv(csv_path, index=False)
