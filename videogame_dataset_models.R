#Import libraries

library(magrittr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(rpart)
library(randomForest)

###Note: Descriptive stats and graphs for Section A is in videogames.R code

#----------------------Analyze your predictors

#Set working directory
setwd('C:/Users/btugc/OneDrive/Documents/Rproject')

#Open data file
videogames <- read.csv2('video_games_dataset.csv')

#Characteristics of the dataset
str(videogames)

#First part of our data
head(videogames)

##----------------------Take the first 500 observation as the training set and use the rest as test set
train <- videogames[1:500,]
test <- videogames[501:nrow(videogames),]

##----------------------Fit a multiple regression model by using only the training set to predict Global_Sales

#Build the Model
model_regression <- lm(train$Global_Sales ~  train$User_Count + train$Critic_Count + train$Critic_Score)

##----------------------Analyze your estimated models, comment on coefficients, adjusted R square and F statistic of the model
summary(model_regression)

##----------------------Predict Global_Sales in the test set using the regression model obtained in (b). 
##----------------------Calculate the root mean square error (RMSE).

#Wrap the predictors
new_data = data.frame(test$User_Count, test$Critic_Score, test$Critic_Count)
predictions_reg <- predict(model_regression, new_data)

#Print Predictions
predictions_reg

#Calculate RMSE
RMSE(test$Global_Sales, predictions_reg)

##----------------------Fit a Ridge model and a Lasso model to predict Global_Sales. 
##----------------------Use only the training set to fit these regression models. 
##----------------------Determine the lambda parameter using cross-validation.


#Set lambda sequences
lambdas <- 10^seq(3, -2, by = -.1)

#Variables
xs <- train %>% select('Critic_Count', 'User_Count', 'Critic_Score') %>% data.matrix()
y <- train$Global_Sales

#--Models
model_ridge <- glmnet(xs, y, alpha = 0, lambda = lambdas)
summary(model_ridge)

model_lasso <- glmnet(xs, y, alpha = 1, lambda = lambdas)
summary(model_lasso)

#--Perform cross-validation

#CV for ridge model
model_ridge_cv <- cv.glmnet(xs, y, alpha = 0, lambda = lambdas, standardize = TRUE, nfolds = 10)
plot(model_ridge_cv)

#Get the best lambda hyperparameter for ridge model
model_ridge_best_lambda <- model_ridge_cv$lambda.min
model_ridge_best_lambda


#CV for lasso model
model_lasso_cv <- cv.glmnet(xs, y, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 10)
plot(model_lasso_cv)
model_lasso_best_lambda <- model_lasso_cv$lambda.min
model_lasso_best_lambda


#--Rebuild the Models with the best lambdas
best_model_ridge <- glmnet(xs, y, alpha = 0, lambda = model_ridge_best_lambda)
summary(best_model_ridge)

best_model_lasso <- glmnet(xs, y, alpha = 1, lambda = model_lasso_best_lambda)
summary(best_model_lasso)

##----------------------Predict Global_Sales in the test set using the Ridge model and the Lasso model obtained in (e). 
##----------------------Calculate RMSEs of these models.
new_x = test %>% select('Critic_Count', 'User_Score', 'Critic_Score') %>% data.matrix()

#Predictions for Ridge Model
predictions_ridge <- predict(best_model_ridge, s = model_ridge_best_lambda, newx = new_x)
predictions_ridge

#RMSE for Ridge Model Predictions
RMSE(test$Global_Sales, predictions_ridge)

#Predictions for Lasso Model
predictions_lasso <- predict(best_model_lasso, s = model_lasso_best_lambda, newx = new_x)
predictions_lasso

#RMSE for Lasso Model Predictions
RMSE(test$Global_Sales, predictions_lasso)

##----------------------Analyze your Lasso Model. Compare your Lasso Model with the multiple regression model

#Get the coefficients of Lasso
coef(best_model_lasso)

#Compare R square

actual <- test$Global_Sales

preds_lasso <- predictions_lasso
rss <- sum((preds_lasso - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
lasso_rsq <- 1 - rss/tss
lasso_rsq

preds_reg <- predictions_reg
rss <- sum((preds_reg - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
reg_rsq <- 1 - rss/tss
reg_rsq

##----------------------Fit a regression tree to predict Global_Sales. 
##----------------------Use only the training set to fit the regression model. 
##----------------------Determine the number of terminal nodes using cross-validation.

# Fit the regression tree
tree_reg <- rpart(train$Global_Sales ~  train$User_Count + train$Critic_Count + train$Critic_Score,
             method="anova")

#Print the steps of splitting
summary(tree_reg)

#Plot the CV results with the size of the tree
plotcp(tree_reg)

#Find optimal cp value
min <- which.min(tree_reg$cptable[, "xerror"])
cp <- tree_reg$cptable[min, "CP"]

#Plot CV results with number of splits
par(mfrow=c(1,2)) # two plots on one page
rsq.rpart(tree_reg)

#Display the tree results
printcp(tree_reg)

#Get the 
# prune the tree
prune_tree_reg <- prune(tree_reg, cp=cp) # from cptable

printcp(prune_tree_reg)

# plot the pruned tree
plot(prune_tree_reg, uniform=TRUE,
     main="Pruned Regression Tree for Sales")
text(prune_tree_reg, use.n=TRUE, all=TRUE, cex=.8)

##----------------------Predict Global_Sales in the test set using the regression tree model obtained in (h). 
##----------------------Calculate the RMSEs of the regression tree.

predictions_regtree <- predict(prune_tree_reg, newdata = new_data)
predictions_regtree

#Calculate RMSE
RMSE(pred = predictions_regtree, obs = test$Global_Sales)

##----------------------Fit random forests to predict Global_Sales. 
##----------------------Use only the training set to fit the regression model. 
##----------------------Determine the number of variables used in each split using the cross-validation. 
##----------------------Grow 500 trees for random forest.

model_forest <- randomForest(Global_Sales ~  User_Count + Critic_Count + Critic_Score, data = train,
                             method = 'rf', ntree = 500)

#See the results
print(model_forest)

#Perform CV

xdata = data.frame(train$User_Count, train$Critic_Score, train$Critic_Count)
rfcv(xdata, train$Global_Sales, cv.fold = 10)

##----------------------Importance of each predictor variable
importance(model_forest) 
varImpPlot(model_forest)


##----------------------Predict Global_Sales in the test set using the random forest model obtained in (j). 
##----------------------Calculate the RMSEs of the random forest
names(new_data)[names(new_data) == 'test.User_Count'] <- 'User_Count'
names(new_data)[names(new_data) == 'test.Critic_Score'] <- 'Critic_Score'
names(new_data)[names(new_data) == 'test.Critic_Count'] <- 'Critic_Count'

predictions_forest <- predict(model_forest, newdata = new_data)
RMSE_forest <- sqrt(sum((predictions_forest - test$Global_Sales)^2)/length(predictions_forest))
print(RMSE_forest)
