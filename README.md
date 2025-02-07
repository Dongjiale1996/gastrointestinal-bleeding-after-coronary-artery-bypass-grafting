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
│   ├── Multilayer_Perceptron/ # Multilayer Perceptron                                                                                                                          
│   ├── Naive_Bayes/         # Naive Bayes                                                                                                                                      
│   ├── Random_Forest/       # Random Forest                                                                                                                                    
│   ├── Support_Vector_Machine/ # Support Vector Machine                                                                                                                        
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
- Mutual Information (MI): Measures the dependency between features and the target variable.
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
To run the code in this project, you need to install several required dependencies. The libraries and frameworks used in the project are listed in the requirements.txt file. You can install all dependencies by running the following command:
pip install scikit-learn==1.0.2; 
pip install pandas==1.3.5; 
pip install matplotlib==3.5.3; 
pip install numpy==1.21.6; 
pip install keras==2.3.1; 
pip install tensorflow==2.2.0; 
pip install imbalanced-learn==0.10.1; 
pip install scipy==1.4.1; 
pip install xgboost==1.6.2; 
pip install joblib==1.2.0; 
pip install seaborn==0.12.2; 
pip install scikit-learn-intelex==2023.2.1; 
pip install sklearnex==0.0.1
4. Usage                                                                                                                                               
Step 1: Data Preprocessing                                                                                                                             
The folder contains scripts for data preprocessing, including normalization and imputing missing values. Run the appropriate scripts for data normalization and imputing missing values.
Example:                                                                                                                                               
Navigate to the folder:                                                                                                                                
cd data_preprocess/                                                                                                                                    
Run the relevant script, such as:                                                                                                                      
python data_normalization.py                                                                                                                           
Step 2: Feature Selection                                                                                                                              
The folder contains scripts for various feature selection methods aimed at reducing model complexity and enhancing performance by retaining only the most relevant features. You can run the scripts to apply the following methods: K-Best method, LASSO method, mutual information-based selection, and Recursive Feature Elimination with Random Forest.
Example:                                                                                                                                               
Navigate to the folder:                                                                                                                                
cd feature_selection/                                                                                                                                  
Run the script to select the most important features, such as:                                                                                         
python k_best.py                                                                                                                                       
Step 3: Model Construction                                                                                                                             
The folder contains scripts to train and evaluate various prediction models. You can choose from machine learning models such as Logistic Regression, Random Forest, XGBoost, Support Vector Machine, Naive Bayes, and Multilayer Perceptron. Each model has a corresponding training script.
Example:                                                                                                                                               
Navigate to the folder:                                                                                                                                
cd model_construction/                                                                                                                                 
Run the corresponding script for the desired model, such as:                                                                                           
python train_logistic_regression.py                                                                                                                    
5. Contact
For any inquiries, please contact:
Email: dongjiale1996@163.com
GitHub: https://github.com/Dongjiale1996/gastrointestinal-bleeding-after-coronary-artery-bypass-grafting
