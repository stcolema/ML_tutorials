

# XGBoost is an implementation of the Gradient Boosted Decision Trees algorithm

# We go through cycles that repeatedly builds new models and combines them into
# an ensemble model. We start the cycle by taking an existing model and
# calculating the errors for each observation in the dataset. We then build a
# new model to predict these errors. We add predictions from this
# error-predicting model to the "ensemble of models."

# === Environment setup ========================================================

# libraries we're going to use

# for xgboost
library(xgboost) # install.packages("xgboost", dep = T)

# general utility functions
library(tidyverse) # install.packages("tidyverse", dep = T)

# convert log odds to probability
odds_to_probs <- function(odds){
  return(exp(odds)/ (1 + exp(odds)))
}


# For this tutorial, we're going to be using a dataset from the Food and
# Agriculture Organization of the United Nations that contains information on
# various outbreaks of animal diseases. We're going to try to predict which
# outbreaks of animal diseases will lead to humans getting sick.

# read in our data & put it in a data frame
diseaseInfo <- read_csv("./input/Outbreak_240817.csv")

# set a random seed & shuffle data frame
set.seed(1234)
diseaseInfo <- diseaseInfo[sample(1:nrow(diseaseInfo)), ]

# One stumbling block when getting started with the xgboost package in R is that
# you can't just pass it a dataframe. The core xgboost function requires data to
# be a matrix. XGBoost has a built-in datatype, DMatrix, that is particularly
# good at storing and accessing sparse matrices efficiently.


# print the first few rows of our dataframe
head(diseaseInfo)

# === Data cleaning ============================================================

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

# get a boolean vector of training labels
diseaseLabels <- diseaseInfo %>%
  select(humansAffected) %>% # get the column with the # of humans affected
  is.na() %>% # is it NA?
  magrittr::not() # switch TRUE and FALSE (using function from the magrittr package)

# check out the first few lines
head(diseaseLabels) # of our target variable
head(diseaseInfo$humansAffected) # of the original column

# select just the numeric columns
diseaseInfo_numeric <- diseaseInfo_humansRemoved %>%
  select(-Id) %>% # the case id shouldn't contain useful information
  select(-c(longitude, latitude)) %>% # location data is also in country data
  select_if(is.numeric) # select remaining numeric columns

# make sure that our dataframe is all numeric
str(diseaseInfo_numeric)

# Convert categorical information (like country) to a numeric format
# Alright, so now we've got all the numeric values we need. But what about
# non-numeric variables? For example, we have a column "country" that tells us
# which country an observation is from.

# check out the first few rows of the country column
head(diseaseInfo$country)

# How can we convert these categories to a matrix? One way to do this is using
# one-hot encoding. One-hot encoding takes each category and makes it its own
# column. Then, for each observation, it puts a "0" in that column if that
# observation doesn't belong to that column and "1" if it does. In R, we can
# convert a column with a categorical variable in it to a one-hot matrix like so:

# one-hot matrix for just the first few rows of the "country" column (note sparisty)
model.matrix(~ country - 1, head(diseaseInfo))

# convert categorical factor into one-hot encoding
region <- model.matrix(~ country - 1, diseaseInfo)


# some of the species
head(diseaseInfo$speciesDescription)

# add a boolean column to our numeric dataframe indicating whether a species is domestic
diseaseInfo_numeric$is_domestic <- str_detect(diseaseInfo$speciesDescription, "domestic")

# get a list of all the species by getting the last
speciesList <- diseaseInfo$speciesDescription %>%
  str_replace("[[:punct:]]", "") %>% # remove punctuation (some rows have parentheses)
  str_extract("[a-z]*$") # extract the least word in each row

# convert our list into a dataframe...
speciesList <- tibble(species = speciesList)

# and convert to a matrix using 1 hot encoding
options(na.action = "na.pass") # don't drop NA values!
species <- model.matrix(~ species - 1, speciesList)

# add our one-hot encoded variable and convert the dataframe into a matrix
diseaseInfo_numeric <- cbind(diseaseInfo_numeric, region, species)
diseaseInfo_matrix <- data.matrix(diseaseInfo_numeric)

# get the numb 70/30 training test split
numberOfTrainingSamples <- round(length(diseaseLabels) * .7)

# training data
train_data <- diseaseInfo_matrix[1:numberOfTrainingSamples, ]
train_labels <- diseaseLabels[1:numberOfTrainingSamples]

# testing data
test_data <- diseaseInfo_matrix[-(1:numberOfTrainingSamples), ]
test_labels <- diseaseLabels[-(1:numberOfTrainingSamples)]

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label = train_labels)
dtest <- xgb.DMatrix(data = test_data, label = test_labels)

# === Training =================================================================

# train a model using our training data
model <- xgboost(
  data = dtrain, # the data
  nround = 2, # max number of boosting iterations
  objective = "binary:logistic"
) # the objective function

# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

# === Tuning our model =========================================================

# Good news! Our initial model had a slightly lower error on our testing data
# than our training data. (That's great, it means we didn't over-fit!)

# Let's imagine, however, that we didn't see a lower error for our training data
# and we were worried that we might be over-fitting.

#     Over-fitting: when your model relies too much on randomness/noise
#     in your training set to make its classifications. As a result, it
#     will probably not extend well to a new dataset.

# One way to avoid over-fitting is to make your model less complex. You can do
# this in xgboost by specifying that you want your decision trees to have fewer
# layers rather than more layers. Each layer splits the remaining data into
# smaller and smaller pieces and therefore makes it more likely that you're
# capturing randomness and not the important variation.

# By default, the max.depth of trees in xgboost is 6. Let's set it to 3 instead.

# train an xgboost model
model_tuned <- xgboost(
  data = dtrain, # the data
  max.depth = 3, # the maximum depth of each decision tree
  nround = 2, # max number of boosting iterations
  objective = "binary:logistic"
) # the objective function

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

# There are two things we can try to see if we improve our model performance.

# 1. Account for the fact that we have imbalanced classes. "Imbalanced classes"
#    just means that we have more examples from one category than the other. In
#    this case, humans don't usually get sick when animals do, but sometimes
#    they do. We can help make sure that we're making sure to predict rare
#    events by scaling the weight we give to positive cases.
#
# 2. Train for more rounds. If we stop training early, it's possible that our
#    error rate is higher than it could be if we just kept at it for a little
#    longer. It's also possible that training longer will result in a more
#    complex model than we need and will cause us to over-fit. We can help
#    guard against this by setting a second parameter, early_stopping_rounds,
#    that will stop training if we have no improvement in a certain number of
#    training rounds.
#
# Let's try re-training our model with these tweaks.

# get the number of negative & positive cases in our data
negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)

# train a model using our training data
model_tuned <- xgboost(
  data = dtrain, # the data
  max.depth = 3, # the maximum depth of each decision tree
  nround = 10, # number of boosting rounds
  early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
  objective = "binary:logistic", # the objective function
  scale_pos_weight = negative_cases / postive_cases
) # control for imbalanced classes

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))



# There are a couple things to notice here.
#
# First, our error in the first round was actually higher than it was for earlier
# models (0.016... vs 0.014...). This is because we've penalized failing to
# capture very rare events.
#
# Then, as we add more training rounds, our error drops a little bit and
# actually ends up lower than it was with our earlier model. This is because
# adding additional training rounds adds additional complexity to our model
# that better allows it to capture the variation in our training data.
#
# After a while, though, our error starts to actually go up. This is probably
# due to over-fitting: we end up at the point where adding more complexity to
# the model is actually hurting it. We've talked about avoiding over-fitting
# above, but another technique that can help avoid over-fitting is adding a
# regularization term, gamma. Gamma is a measure of how much an additional split
# will need to reduce loss in order to be added to the ensemble. If a proposed
# model does not reduce loss by at least whatever-you-set-gamma-to, it won't be
# included. Here, I'll set it to one, which is fairly high. (By default gamma
# is zero.)

# train a model using our training data
model_tuned <- xgboost(
  data = dtrain, # the data
  max.depth = 3, # the maximum depth of each decision tree
  nround = 10, # number of boosting rounds
  early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
  objective = "binary:logistic", # the objective function
  scale_pos_weight = negative_cases / postive_cases, # control for imbalanced classes
  gamma = 1
) # add a regularization term

# generate predictions for our held-out testing data
pred <- predict(model_tuned, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

# Adding a regularization terms makes our model more conservative, so it doesn't
# end up adding the models which were reducing our accuracy.

# === Examining the model ======================================================

# plot them features! what's contributing most to our model?
xgb.plot.multi.trees(feature_names = names(diseaseInfo_matrix), 
                     model = model)




# The top of the tree is on the left and the bottom of the tree is on the right.
# For features, the number next to it is "quality", which helps indicate how 
# important this feature was across all the trees. Higher quality means a 
# feature was more important. So we can tell that is_domestic was by far the
# most important feature across all of our trees, both because it's higher in
# the tree and also because it's quality score is very high.
# 
# For the nodes with "Leaf", the number next to the "Leaf" is the average value 
# the model returned across all trees if a a certain observation ended up in 
# that leaf. Because we're using a logistic model here, it's telling us the 
# log-odds rather than the probability. We can pretty easily convert the log 
# odds to probability, though.

# probability of leaf above countryPortugul
odds_to_probs(-0.599)


# So, in the trees where an observation ended up in that leaf, on average the 
# probability that a human would be sick in that instance was 35%. Since that 
# was below the threshold of 50% we used for our decision rule, we'd say that 
# these instance usually wouldn't result in a human getting sick.

# What if we want a quick way to see which features are most important? We can 
# do that using by creating and then plotting the importance matrix, like so:
  
# get information on how important each feature is
importance_matrix <- xgb.importance(names(diseaseInfo_matrix), model = model)

# and plot it!
xgb.plot.importance(importance_matrix)



# Here, each bar is a different feature, and the x-axis is plotting the weighted 
# gain. ("Weighted" just means that if you add together the gain for every 
# feature, you'll get 1.) Basically this plot tells us how informative each
# feature was when we look at every tree in our ensemble. So features with a
# lot of gain were very important to our model while features with less gain
# were less helpful.

# Here, we can see that whether or not an animal that got sick was domestic was 
# far and away the most important feature for figuring out if humans would get 
# sick! This makes sense, since a person is more likely to come in contact with
# a domestic animal than a wild one.
