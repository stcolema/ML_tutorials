

# XGBoost is an implementation of the Gradient Boosted Decision Trees algorithm

# We go through cycles that repeatedly builds new models and combines them into
# an ensemble model. We start the cycle by taking an existing model and 
# calculating the errors for each observation in the dataset. We then build a 
# new model to predict these errors. We add predictions from this 
# error-predicting model to the "ensemble of models."

# libraries we're going to use

# for xgboost
library(xgboost) # install.packages("xgboost", dep = T)

# general utility functions
library(tidyverse) # install.packages("tidyverse", dep = T)

# For this tutorial, we're going to be using a dataset from the Food and 
# Agriculture Organization of the United Nations that contains information on 
# various outbreaks of animal diseases. We're going to try to predict which 
# outbreaks of animal diseases will lead to humans getting sick.

# read in our data & put it in a data frame
diseaseInfo <- read_csv("./input/Outbreak_240817.csv")

# set a random seed & shuffle data frame
set.seed(1)
diseaseInfo <- diseaseInfo[sample(1:nrow(diseaseInfo)), ]

# One stumbling block when getting started with the xgboost package in R is that 
# you can't just pass it a dataframe. The core xgboost function requires data to
# be a matrix. XGBoost has a built-in datatype, DMatrix, that is particularly 
# good at storing and accessing sparse matrices efficiently.


# print the first few rows of our dataframe
head(diseaseInfo)

# Necessary cleaning:
#   Remove information about the target variable from the training data
#   Reduce the amount of redundant information
#   Convert categorical information (like country) to a numeric format
#   Split dataset into testing and training subsets
#   Convert the cleaned dataframe to a Dmatrix

# get the subset of the dataframe that doesn't have labels about 
# humans affected by the disease
diseaseInfo_humansRemoved <- diseaseInfo %>%
  select(-starts_with("human"))





