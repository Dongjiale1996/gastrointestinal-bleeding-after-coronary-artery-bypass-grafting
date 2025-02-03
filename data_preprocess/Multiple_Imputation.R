# Load necessary libraries
library(lattice)  # Load the lattice library for graphical visualizations
library(MASS)     # Load the MASS library for statistical functions and datasets
library(nnet)     # Load the nnet library for neural network modeling
library(mice)     # Load the mice library for multiple imputation of missing data

# Load the dataset
data <- read.csv("E:\\Ashuju\\xin43\\With_missing_data.csv")  # Read the dataset from the specified file path

# View data structure and summary
View(data)  # Open the dataset in a viewer
ca <- data  # Create a copy of the dataset
dim(ca)  # Check the dimensions (rows and columns) of the dataset
str(ca)  # Display the structure of the dataset (variable types, etc.)
sum(is.na(ca))  # Count the total number of missing values in the dataset

# Perform multiple imputation
imp <- mice(ca, seed = 1234)  # Perform multiple imputations (default is 5 datasets) using Monte Carlo simulation

# Fit a generalized linear model (GLM) on each imputed dataset
fit <- with(imp, glm(label ~ ., data = ca))  # Fit a GLM with `label` as the dependent variable on the imputed datasets

# Pool results from the fitted models
pool <- pool(fit)  # Combine results across the 5 imputed datasets to produce pooled estimates

# Set display options and show summary of the pooled results
options(digits = 2)  # Set numerical output to 2 decimal places
summary(pool)  # Display summary statistics of the pooled results (e.g., coefficients, standard errors)

# View imputed values for a specific variable
imp$imp$clac  # View the imputed values for the variable `clac`
imp$method    # View the imputation method used for each variable

# Extract one of the imputed datasets
ca_complete <- complete(imp, action = 3)  # Retrieve the third imputed dataset (action = 3)

# Check for missing values in the completed dataset
sum(is.na(ca_complete))  # Verify that no missing values remain in the completed dataset

# Optional visualization (commented out)
# par(mfrow = c(3, 3))  # Set graphical layout to a 3x3 grid
# stripplot(imp, pch = c(1, 8), col = c("grey", "black"))  # Visualize the distribution of observed vs. imputed values
# par(mfrow = c(1, 1))  # Reset the graphical layout to a single plot (1x1)

# Save the completed imputed dataset to a file
write.csv(ca_complete, file = "E:\\Ashuju\\xin43\\Imputed_data.csv")  # Save the imputed dataset to the specified file path
