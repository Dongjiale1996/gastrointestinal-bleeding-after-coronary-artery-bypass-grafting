Prediction Model for Gastrointestinal Bleeding After Coronary Artery Bypass Grafting
1. Overview
This repository contains code to develop a prediction model for gastrointestinal bleeding (GIB) following coronary artery bypass grafting (CABG). Using data from four hospitals and the Medical Information Mart for Intensive Care IV (MIMIC-IV), we constructed and compared various machine learning and deep learning algorithms to select the optimal model for identifying patients at higher risk of experiencing GIB after CABG. The goal of this project is to assist clinicians in making more informed decisions, reduce complications, and improve patient care by predicting the likelihood of GIB in patients post-surgery.
The project structure is as follows:
├── data_preprocess/          # Data preprocessing scripts

│   ├── Data_Normalization/  # Data normalization

│   ├── Multiple_Imputation/ # Data imputation

│   ├── Sample_size/         # Sample size analysis

├── feature_selection/        # Feature selection scripts

│   ├── K_Best/              # K-Best feature selection

│   ├── LASSO/               # LASSO feature selection

│   ├── Mutual_Information/  # Mutual information feature selection

│   ├── RFE_RF/              # Recursive Feature Elimination with Random Forest

├── model_construction/       # Model construction scripts
│   ├── Logistic_Regression/ # Logistic Regression

│   ├── Multilayer_Perceptron/ # Multilayer Perceptron (MLP)

│   ├── Naive_Bayes/         # Naive Bayes
│   ├── Random_Forest/       # Random Forest
│   ├── Support_Vector_Machine/ # Support Vector Machine (SVM)
│   ├── XGBoost/             # XGBoost
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
3. Methods
2.1 Data Preprocessing
The dataset underwent rigorous preprocessing, including multiple imputation for handling missing data and normalization to standardize feature scales. These steps ensured the data was suitable for model training and evaluation.
2.2 Feature Selection
To enhance model performance, we employed several feature selection techniques:
- K-Best: Selects features based on statistical tests.
- LASSO: Uses L1 regularization to identify the most relevant features.
- Mutual Information: Measures the dependency between features and the target variable.
- Recursive Feature Elimination with Random Forest (RFE-RF): Iteratively removes the least important features based on Random Forest importance scores.
2.3 Model Construction
We implemented and compared six supervised machine learning algorithms:
- Logistic Regression (LR): A statistical method for binary classification, predicting the probability of an event.
- Random Forest (RF): An ensemble method that aggregates predictions from multiple decision trees.
- XGBoost: A scalable gradient boosting framework optimized for performance and accuracy.
- Support Vector Machine (SVM): Effective for high-dimensional spaces, suitable for classification tasks.
- Naive Bayes (NB): A probabilistic classifier based on Bayes' theorem with strong independence assumptions.
- Multilayer Perceptron (MLP): A feedforward neural network for supervised learning tasks.
2.4 Hyperparameter Optimization
Each model underwent hyperparameter tuning using five-fold grid search cross-validation. This process systematically explored hyperparameter combinations to identify the optimal configuration based on performance metrics.
2.5 Model Evaluation
The final model selection was based on a comprehensive evaluation of the following metrics:
- Brier Score: Measures the accuracy of predicted probabilities (lower values indicate better performance).
- AUC-ROC: Evaluates the model's ability to distinguish between classes (higher values indicate better performance).
3. Requirements
To run the code in this project, you need to install several required dependencies. The libraries and frameworks used in the project are listed in the `requirements.txt` file. You can install all dependencies by running the following command:
```bash
pip install -r requirements.txt
```
4. Usage
Step 1: Data Preprocessing
The folder contains scripts for preprocessing data. This step involves normalizing and imputing missing values. You need to run the following scripts:
- In the folder, run the script for data normalization (`Data_Normalization/`).
- In the folder, run the script for imputing missing values (`Multiple_Imputation/`).
Example:
1. Navigate to the folder: `data_preprocess/`
2. Run the relevant script, such as:
   ```bash
   python data_normalization.py
   ```
Step 2: Feature Selection
In the folder, you will find scripts for different feature selection methods. Feature selection helps to reduce the complexity of your model and improve its performance by retaining only the most relevant features:
- Run the scripts in the folder to apply the K-Best method (`K_Best/`).
- Run the scripts in the folder to apply the LASSO method (`LASSO/`).
- Run the scripts in the folder for mutual information-based selection (`Mutual_Information/`).
- Run the scripts in the folder for Recursive Feature Elimination with Random Forest (`RFE_RF/`).
Example:
1. Navigate to the folder: `feature_selection/`
2. Run the script to select the most important features:
   ```bash
   python k_best.py
   ```
Step 3: Model Construction
The folder contains scripts to train and evaluate prediction models. You can choose from various machine learning models such as Logistic Regression, Random Forest, XGBoost, Support Vector Machine, Naive Bayes, and Multilayer Perceptron. Each of these models has a corresponding script for training.
1. Navigate to the `model_construction/` folder.
2. Run the corresponding script for the desired model:
   - Logistic Regression: `train_logistic_regression.py`
   - Multilayer Perceptron (MLP): `train_multilayer_perceptron.py`
   - Naive Bayes: `train_naive_bayes.py`
   - Random Forest: `train_random_forest.py`
   - Support Vector Machine (SVM): `train_support_vector_machine.py`
   - XGBoost: `train_xgboost.py`
These scripts will train the model, generate predictions, and evaluate performance based on metrics like AUC and Brier score.
5. Contact
For any inquiries, please contact:
Email: dongjiale1996@163.com
GitHub: https://github.com/Dongjiale1996/gastrointestinal-bleeding-after-coronary-artery-bypass-grafting
