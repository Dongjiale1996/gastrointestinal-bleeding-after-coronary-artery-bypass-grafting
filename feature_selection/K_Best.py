# Import necessary libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
dataTrain = 'E:\\Ashuju\\xin43\\derivation_nm.csv'
data_Train = pd.read_csv(dataTrain)

# Split data into features (X) and target variable (Y)
X = data_Train.iloc[:, 0:-1]
Y = data_Train['label']

# Perform feature selection
KB = SelectKBest(f_classif, k=34)
X_new = KB.fit_transform(X, Y)
scores = KB.scores_

# Sort features by their scores in descending order
indices = np.argsort(scores)[::-1]
k_best_list = [X.columns[indices[i]] for i in range(34)]

# Plot the feature score distribution chart
scores_list = abs(np.sort(-KB.scores_))
x = range(1, len(scores_list) + 1)
plt.figure(dpi=300)
plt.plot(x, scores_list)
plt.xlabel("Feature Index")
plt.ylabel("Score")
plt.grid(ls=':', color='gray')
plt.show()

# Create a DataFrame to display features and their scores
df_feature_scores = pd.DataFrame({
    'Feature': k_best_list,
    'Score': scores_list
})
df_feature_scores.index = np.arange(1, len(df_feature_scores) + 1)
print(df_feature_scores)

#  Select the score threshold based on the feature score distribution chart.
#  In this study, features with retention scores greater than 100, corresponding to the top 16 features, were selected.
df_feature_scores_top = df_feature_scores.iloc[:16]

# Save the top features and their scores to a CSV file
csv_path = 'E:\\Ashuju\\xin43\\Features_KB.csv'
df_feature_scores_top.to_csv(csv_path, index=False)
