### MIS-315, Deniz Yucel, 111121003 ###

#Import the necessary libraries

library(magrittr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(tree)
library(randomForest)
library(gdata)
library(Metrics)
library(corrplot)
library(car)


#----------------------First drop all goalkeepers from the dataset. If a player has a gk attribute, then it means he is a goalkeeper.

#Set working directory
setwd('C:/Users/btugc/OneDrive/Documents/Rproject/')

#Open data file
fifa <- read.csv2('Final_Project_Data_Set(1).csv', header = TRUE, sep = ',')

#Characteristics of the dataset
str(fifa)

#First part of our data
head(fifa)

reg_fifa <- data.frame()

#Drop the goal keepers
removed_row_list <- c()

for (row in 1:1599)
  if (is.na(fifa[row, 38]) == FALSE)
  {
    removed_row_list <- cbind(removed_row_list, row)
  }
    
reg_fifa <- fifa[ -removed_row_list, ]

#Remove the goal keeper column since it is only values of NA
#also the ID and Name

reg_fifa <- reg_fifa[, 3:37]

##----------------------Analyze your predictors

summary(reg_fifa)
str(reg_fifa)
head(reg_fifa)

##--Correlation Plot
num_reg_fifa <- reg_fifa[ -c(4, 5, 6)]
res <- cor(num_reg_fifa[, 20:23])
round(res, 2)
corrplot(res, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)
pairs(num_reg_fifa[, 29:32])


##----------------------Take the first 1000 observation as the training set and use the rest as test set
train <- reg_fifa[1:1000,]
test <- reg_fifa[1001:nrow(reg_fifa),]

##----------------------Fit a multiple regression model by using only the training set to predict eur_value

#Build the Model
model_regression <- lm(eur_value ~ ., train)

##----------------------Analyze your estimated models, comment on coefficients, adjusted R square and F statistic of the model
summary(model_regression)
coef(model_regression)
range(reg_fifa$eur_value)
vif(model_regression)


##----------------------Predict "eur_value" in the test set using the regression model obtained in (d). 
##----------------------Calculate the mean square error of the test set (MSE).

#Wrap the predictors
new_data = data.frame(test)
predictions_reg <- predict(model_regression, new_data, interval = 'confidence')

#Print Predictions
predictions_reg

#Calculate MSE
mse(test$eur_value, predictions_reg)

##----------------------Fit a Ridge model and a Lasso model to predict eur_value. 
##----------------------Use only the training set to fit these regression models. 
##----------------------Determine the lambda parameter using cross-validation.


#Set lambda sequences
lambdas <- 10 ^ seq(10, -2, length = 100)

#Variables
xs <- model.matrix(eur_value ~ ., data = train)[,-1]
y <- train$eur_value

#--Models
model_ridge <- glmnet(xs, y, alpha = 0, lambda = lambdas)
summary(model_ridge)

model_lasso <- glmnet(xs, y, alpha = 1, lambda = lambdas)
summary(model_lasso)

#--Perform cross-validation

#CV for ridge model
model_ridge_cv <- cv.glmnet(xs, y, lambda = lambdas, nfolds = 5)
plot(model_ridge_cv)

#Get the best lambda hyperparameter for ridge model
model_ridge_best_lambda <- model_ridge_cv$lambda.min
model_ridge_best_lambda
log(model_ridge_best_lambda)

#CV for lasso model
model_lasso_cv <- cv.glmnet(xs, y, alpha = 1, lambda = lambdas, nfolds = 5)
plot(model_lasso_cv)
model_lasso_best_lambda <- model_lasso_cv$lambda.min
model_lasso_best_lambda
log(model_lasso_best_lambda)


#--Rebuild the Models with the best lambdas
best_model_ridge <- glmnet(xs, y, alpha = 0, lambda = model_ridge_best_lambda)
summary(best_model_ridge)

best_model_lasso <- glmnet(xs, y, alpha = 1, lambda = model_lasso_best_lambda)
summary(best_model_lasso)

#Plot coefficients vs log-lambda for Lasso Regression (Figure 6)
plot(model_lasso, "lambda", label=TRUE)
coef(best_model_lasso)

##----------------------Predict "eur_value" in the test set using the Ridge model and the Lasso model obtained in (g). 
##----------------------Calculate MSEs of these models.
new_x <- model.matrix(eur_value ~ ., data = test)[,-1]

#Predictions for Ridge Model
predictions_ridge <- predict(best_model_ridge, s = model_ridge_best_lambda, newx = new_x)
predictions_ridge

#MSE for Ridge Model Predictions
mse(test$eur_value, predictions_ridge)

#Predictions for Lasso Model
predictions_lasso <- predict(best_model_lasso, s = model_lasso_best_lambda, newx = new_x)
predictions_lasso

#MSE for Lasso Model Predictions
mse(test$eur_value, predictions_lasso)



##----------------------Fit a regression tree to predict eur_value. 
##----------------------Use only the training set to fit the regression model. 
##----------------------Determine the number of terminal nodes using cross-validation.

# Fit the regression tree
set.seed(1)
tree_reg <- tree(eur_value ~., train, subset = xs)

#Print the steps of splitting
summary(tree_reg)

#Cross Validation
cv_tree_reg <- cv.tree(tree_reg)
plot(cv_tree_reg$size, cv_tree_reg$dev, type='b')

#Prune the tree
prune_tree_reg <- prune.tree(tree_reg, best=8)
plot(prune_tree_reg, 
     main = 'Prunned Tree')
text(prune_tree_reg, pretty=0)


##----------------------Predict eur_value in the test set using the regression tree model obtained in (h). 
##----------------------Calculate the MSEs of the regression tree.

predictions_regtree <- predict(prune_tree_reg, newdata = new_data)
predictions_regtree

#Calculate MSE
mse(predictions_regtree, test$eur_value)

##----------------------Fit random forests to predict eur_value. 
##----------------------Use only the training set to fit the regression model. 
##----------------------Determine the number of variables used in each split using the cross-validation. 
##----------------------Grow 500 trees for random forest.

set.seed(1)
train.mat <- model.matrix(eur_value ~ ., data = train)
#Perform CV
rfcv(train[,2:35], train[,1], cv.fold=10, scale="log", step=0.5,  mtry=function(p) max(1, floor(sqrt(p))), recursive=FALSE)

model_forest <- randomForest(eur_value ~ ., data = train, subset = train.mat, mtry = 17, ntree = 500, importance = TRUE, na.action = na.omit)


#See the results
print(model_forest)


##----------------------Importance of each predictor variable
importance(model_forest)
varImpPlot(model_forest)
library(caret)
varImp(model_forest)
varImpPlot(model_forest,type=2)


##----------------------Predict Global_Sales in the test set using the random forest model obtained in (j). 
##----------------------Calculate the MSEs of the random forest

names(new_data)[names(new_data) == 'test.composure'] <- 'composure'
names(new_data)[names(new_data) == 'test.reactions'] <- 'reactions'

predictions_forest <- predict(model_forest, newdata = new_data)
predictions_forest
mse(predictions_forest, test$eur_value)

