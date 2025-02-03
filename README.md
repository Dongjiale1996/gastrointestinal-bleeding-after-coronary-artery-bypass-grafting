Prediction Model for Gastrointestinal Bleeding After Coronary Artery Bypass Grafting
Overview
This repository contains the code to develop a prediction model for gastrointestinal bleeding (GIB) after coronary artery bypass grafting (CABG). Using multi-center data from hospitals and healthcare institutions, we employed both machine learning and deep learning techniques to construct a predictive model aimed at identifying patients who are at a higher risk of experiencing gastrointestinal bleeding following CABG surgery. The goal of this project is to help clinicians make more informed decisions, reduce complications, and improve patient care by predicting the likelihood of GIB in patients post-surgery.
Project Structure
The project directory is structured as follows:
```
├── data_preprocess/          # Data preprocessing folder
│   ├── Data_Normalization/  # Data normalization scripts
│   ├── Multiple_Imputation/ # Data imputation scripts
│   ├── Sample_size/         # Sample size analysis scripts
├── feature_selection/        # Feature selection folder
│   ├── K_Best/              # K-Best feature selection scripts
│   ├── LASSO/               # LASSO feature selection scripts
│   ├── Mutual_Information/  # Mutual information feature selection scripts
│   ├── RFE_RF/              # Recursive Feature Elimination with Random Forest scripts
├── model_construction/       # Model construction folder
│   ├── Logistic_Regression/ # Logistic Regression model scripts
│   ├── Multilayer_Perceptron/ # Multilayer Perceptron model scripts
│   ├── Naive_Bayes/         # Naive Bayes model scripts
│   ├── Random_Forest/       # Random Forest model scripts
│   ├── Support_Vector_Machine/ # Support Vector Machine model scripts
│   ├── XGBoost/             # XGBoost model scripts
├── README.md                # Project documentation
```
Methods
In this project, we applied a variety of machine learning and deep learning algorithms to build the prediction model:
- **Logistic Regression (LR)**: A statistical method for binary classification, predicting the probability of the occurrence of an event.
- **Random Forest (RF)**: An ensemble learning method that constructs multiple decision trees and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
- **XGBoost**: A scalable, highly efficient gradient boosting library used for supervised learning tasks.
- **Support Vector Machine (SVM)**: A supervised machine learning algorithm for classification and regression tasks, particularly useful for high-dimensional spaces.
- **Naive Bayes (NB)**: A probabilistic classifier based on applying Bayes’ theorem with strong (naive) independence assumptions between the features.
- **Multilayer Perceptron (MLP)**: A type of feedforward artificial neural network for supervised learning tasks, using a series of interconnected layers of neurons.

The data was preprocessed using a combination of normalization and imputation techniques. Features were selected using various methods such as K-Best, LASSO, mutual information, and Recursive Feature Elimination with Random Forest. These methods helped to improve the predictive power of the models by reducing dimensionality and retaining only the most relevant features.

Model performance was evaluated using several metrics, including:
- **Accuracy**: The proportion of correct predictions made by the model.
- **Brier-score**: A measure of how close the predicted probabilities are to the true outcomes, with lower values indicating better performance.
- **AUC-ROC**: The area under the Receiver Operating Characteristic curve, indicating how well the model distinguishes between classes.
Requirements
To run the code, you need to install the required dependencies. You can install them using the following command:
```bash
pip install -r requirements.txt
```
Alternatively, you can manually install the dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow
```
Usage
The repository contains three main scripts for preprocessing data, selecting features, and training models. Follow the steps below to run the code:

1. **Data Processing**: Preprocess the data by normalizing it, imputing missing values, and performing sample size analysis.
```bash
python scripts/data_processing.py
```
2. **Feature Selection**: Select important features using techniques like K-Best, LASSO, and RFE-RF.
```bash
python scripts/feature_selection.py
```
3. **Model Training**: Train the selected models using the preprocessed data and chosen features.
```bash
python scripts/model_training.py
```
Results
The performance of each model was evaluated using the following metrics:
- **Accuracy**: The percentage of correct predictions among the total predictions.
- **AUC-ROC**: A performance measurement for classification problems at various thresholds settings.
- **Brier-score**: A measure of how close the predicted probabilities are to the true outcomes, with lower values indicating better performance.

The models demonstrated varying performance, with XGBoost achieving the highest AUC-ROC score, indicating its superior performance for this task. Random Forest and Logistic Regression also performed well, offering a good balance between interpretability and predictive accuracy.
Contact
For any questions, feel free to contact:

Email: dongjiale1996@163.com
GitHub: https://github.com/Dongjiale1996/gastrointestinal-bleeding-after-coronary-artery-bypass-grafting/new/master?filename=README.md
