# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load Derivation Cohort
train = 'E:\\Ashuju\\xin38\\data\\derivation_lasso.csv'
train = pd.read_csv(train)
X_train = train.iloc[: ,0:15]
X_train

# Load Drum Tower Validation Cohort
test = 'E:\\Ashuju\\xin38\\data\\validation1_lasso.csv'
test = pd.read_csv(test)
X_test = test.iloc[: ,0:15]
X_test

# Load MIMIC Validation Cohort
test2 = 'E:\\Ashuju\\xin38\\data\\validation2_lasso.csv'
test2 = pd.read_csv(test2)
X_test2 = test2.iloc[: ,0:15]
X_test2

scaler_mm = MinMaxScaler()
#Normalize the Derivation Cohort
scaler_mm_train = scaler_mm.fit(X_train)
scaler_mm
X_train_mm = scaler_mm_train.transform(X_train)
X_train_mm = pd.DataFrame(X_train_mm,columns = X_train.columns)
X_train_mm

#Normalize the Drum Tower Validation Cohort
X_test_mm = scaler_mm_train.transform(X_test)
X_test_mm = pd.DataFrame(X_test_mm,columns = X_test.columns)
X_test_mm

# Normalize the MIMIC Validation Cohort
X_test_mm2 = scaler_mm_train.transform(X_test2)
X_test_mm2 = pd.DataFrame(X_test_mm2,columns = X_test2.columns)
X_test_mm2

# Get label column of Derivation Cohort
train = 'E:\\Ashuju\\xin38\\data\\derivation_lasso.csv'
train = pd.read_csv(train)
X_train_label = train.iloc[: ,15]
X_train_mm['label'] = X_train_label
X_train_mm

# Get label column of Drum Tower Validation Cohort
test = 'E:\\Ashuju\\xin38\\data\\validation1_lasso.csv'
test = pd.read_csv(test)
X_test_label = test.iloc[: ,15]
X_test_mm['label'] = X_test_label
X_test_mm

# Get label column of MIMIC Validation Cohort
test2 = 'E:\\Ashuju\\xin38\\data\\validation2_lasso.csv'
test2 = pd.read_csv(test2)
X_test_label2 = test2.iloc[: ,15]
X_test_mm2['label'] = X_test_label2
X_test_mm2

#Save Data
X_train_mm.to_csv('E:\\Ashuju\\xin38\\data\\derivation_lasso_nm.csv',index=False)
X_test_mm.to_csv('E:\\Ashuju\\xin38\\data\\validation1_lasso_nm.csv',index=False)
X_test_mm2.to_csv('E:\\Ashuju\\xin38\\data\\validation2_lasso_nm.csv',index=False)