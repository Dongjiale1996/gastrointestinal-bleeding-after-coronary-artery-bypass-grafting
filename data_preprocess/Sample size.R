
# Enter estimated incidence 
prevalence <- 0.06  
# Compute max(R²_cs) using the formula
# lnL_null: log-likelihood of an intercept-only model
lnL_null <- prevalence * log(prevalence) + (1 - prevalence) * log(1 - prevalence)
max_R2_cs <- 1 - exp(2 * lnL_null)


# Output the calculated max(R²_cs) value
cat("The calculated max(R²_cs) value is:", max_R2_cs, "\n")

# If we conservatively assume that the new model will explain 15% of the variability,
# the expected R²CS value is calculated as:
rsquared <- 0.15 * max_R2_cs  # Expected R²CS value based on the assumption

# Output the expected R²CS value
cat("The expected R²CS value (rsquared) is:", rsquared, "\n")

# Load necessary library
library(pmsampsize)

# Use pmsampsize function to calculate the sample size
pmsampsize(
  type = "b",         # 'b' corresponds to binary outcomes
  rsquared = rsquared, # Use the expected R²CS value
  parameters = 15,     # Number of candidate predictors
  prevalence = prevalence, # Prevalence of the outcome in the population
  seed = 123456        # Set random seed
)



