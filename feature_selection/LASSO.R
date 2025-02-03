# Load necessary libraries
library(readr)
library(glmnet)  
library(caret)

# Read the dataset from the specified file path
data <- read_csv("E:\\Ashuju\\xin43\\derivation_nm.csv")
data <- na.omit(data)  # Remove rows with missing values

# View the dataset
View(data)

# Set up the independent variables (features) as a matrix
x <- as.matrix(data[, 1:34])  
# Extract the dependent variable (response) from the 35th column
y <- as.matrix(data[, 35])  

# Perform Lasso regression
set.seed(123)  # Set a random seed for reproducibility
lasso <- glmnet(x, y, family = "binomial", alpha = 1)  # Lasso with logistic regression (binomial family)
print(lasso)  # Display the Lasso model details
plot(lasso, xvar = "lambda", label = TRUE)  # Plot Lasso coefficients against log(lambda)

# Perform cross-validated Lasso for variable selection
fitCV <- cv.glmnet(x, y, family = "binomial", type.measure = "deviance", nfolds = 5, alpha = 1)  # 5-fold cross-validation

# Plot the cross-validation results
plot(fitCV)

# Select the largest lambda within one standard error of the minimum deviance
fitCV$lambda.1se  

# View the coefficients of selected variables at the optimal lambda
coef(fitCV, s = "lambda.1se")  

