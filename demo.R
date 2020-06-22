library(tidyverse)
library(caret)
source("ordinal_classification.R")


X = read_csv("data/downsampled_features.csv")
y = read_csv("data/downsampled_target.csv")

shuffle = sample(nrow(X))
data = cbind(y$cluster, X)[shuffle,-2]

trainIndex = createDataPartition(c(1:nrow(X)),p=0.75,list=FALSE)
# trainIndex = sample(c(1:nrow(X)), 0.75 * nrow(X))

# trainX = X[trainIndex,-1]
# testX = X[-trainIndex,-1]
# trainY = y$cluster[trainIndex]# %>% as.factor()
# testY = y$cluster[-trainIndex]# %>% as.factor()

trainX = data[trainIndex, -1]
testX = data[-trainIndex, -1]
trainY = data[trainIndex,1]
testY = data[-trainIndex,1]

# convert y to ordinal matrix
y_encode = ordinal_coder(trainY)

#cross-validation setting
ctrl = trainControl(method = "cv", number = 5, summaryFunction = defaultSummary)


######### K-nearest neighbors with ordinality ################
model_array = c()
for (i in 1:ncol(y_encode)) {
  model_array[[i]] <- train(trainX,
                            y = as.factor(y_encode[,i]),
                            method = "knn",
                            metric = "kappa",
                            preProc = c("center", "scale", "spatialSign", "pca"),
                            tuneGrid = data.frame(.k = 1:15),
                            trControl = ctrl)
}

# knn training set accuracy
train_y_pred = predict_ordinal(trainX, model_array)
mean(trainY == train_y_pred)

# knn test set accuracy
y_pred = predict_ordinal(testX, model_array)
mean(testY == y_pred)


#################### XGBoost with ordinality #########################

xgbGrid <- expand.grid(nrounds = c(50,100),  # this is n_estimators in the python code above
                       max_depth = c(5,7,10),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)
model_array = c()
for (i in 1:ncol(y_encode)) {
  model_array[[i]] <- train(x = trainX, y = as.factor(y_encode[,i]),
                            method = "xgbTree",
                            tuneGrid = xgbGrid,
                            trControl = ctrl,
                            verbose=TRUE)
}

# xgboost training set accuracy
train_y_pred = predict_ordinal(trainX, model_array)
mean(trainY == train_y_pred)

# xgboost test set accuracy
y_pred = predict_ordinal(testX, model_array)
mean(testY == y_pred)


#################### XGBoost no ordinality #########################
xg = train(x = trainX, y = as.factor(trainY),
      method = "xgbTree",
      tuneGrid = xgbGrid,
      trControl = ctrl,
      verbose=TRUE)
# xgboost training set accuracy
train_y_pred = predict(xg, trainX)
mean(trainY == train_y_pred)

# xgboost test set accuracy
y_pred = predict(xg, testX)
mean(testY == y_pred)


################ Logistic regression with ordinality ##################
model_array = c()
for (i in 1:ncol(y_encode)) {
  model_array[[i]] <- train(trainX,
                            y = as.factor(y_encode[,i]),
                            method = "glm",
                            family = "binomial",
                            metric = "accuracy",
                            preProcess = "pca",
                            trControl = ctrl)
}

# xgboost training set accuracy
train_y_pred = predict_ordinal(trainX, model_array)
mean(trainY == train_y_pred)

# xgboost test set accuracy
y_pred = predict_ordinal(testX, model_array)
mean(testY == y_pred)


########################### other models #############################
# logistic regression
library(MASS)
lr = polr(as.factor(trainY) ~ ., data = cbind(trainX, trainY), method="probit")

library(corrplot)
corrplot(cbind(trainY, trainX))

#K-Nearest Neighbors without ordinality
set.seed(1)
knnFit <- train(trainX,
                y = as.factor(trainY),
                method = "knn",
                metric = "kappa",
                preProc = c("center", "scale", "spatialSign", "pca"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.k = 1:15),
                trControl = ctrl)
#knnFit

y_pred = predict(knnFit, testX)
mean(testY == y_pred)

nbaKNNPred = data.frame(obs = as.factor(testY), pred = predict(knnFit, testX))
confusionMatrix(data = nbaKNNPred$pred, reference = nbaKNNPred$obs)
plot(knnFit, main = "Model training for k-nearest neighbors")

pred = predict(knnFit, testX, type="prob")

# knn within loop
train(trainX,
      y = as.factor(y_encode[,i]),
      method = "knn",
      metric = "kappa",
      preProc = c("center", "scale", "spatialSign", "pca"),
      tuneGrid = data.frame(.k = 1:15),
      trControl = ctrl)



#logistic regression no ordinality
set.seed(1)
nbaLog <- train(trainX,
                y = as.factor(trainY),
                method = "multinom",
                metric = "Kappa",
                preProcess = "pca",
                trControl = ctrl)
#nbaLog

# logistic regression training set accuracy
train_y_pred = predict(nbaLog, trainX)
mean(trainY == train_y_pred)

# logistic regression test set accuracy
y_pred = predict(nbaLog, testX)
mean(testY == y_pred)

# nbaLogPred = data.frame(obs = testY, pred = predict(nbaLog, testX))
# confusionMatrix(data = nbaLogPred$pred, reference = nbaLogPred$obs)



#Support Vector Machines
library(kernlab)
sigmaRangeReduced <- sigest(as.matrix(trainX))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-2, 8)))
svmctrl = trainControl(method = "cv", number = 3, summaryFunction = defaultSummary, classProbs = FALSE)
set.seed(1)
svmRModel <- train(trainX,
                   y = trainY,
                   method = "svmRadial",
                   metric = "Kappa",
                   preProc = c("center", "scale", "pca"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = svmctrl)
svmRModel

nbaSVMPred = data.frame(obs = testY, pred = predict(svmRModel, testX))
confusionMatrix(data = nbaSVMPred$pred, reference = nbaSVMPred$obs)
plot(svmRModel, main = "Model training for SVM")

# ordinal random forest
library(ordinalForest)
set.seed(1)
knnFit <- train(trainX,
                y = trainY,
                method = "ordinalForest",
                metric = "Kappa",
                preProc = c("center", "scale", "spatialSign", "pca"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.k = 1:20),
                trControl = ctrl)
knnFit

nbaKNNPred = data.frame(obs = testY, pred = predict(knnFit, testX))
confusionMatrix(data = nbaKNNPred$pred, reference = nbaKNNPred$obs)
plot(knnFit, main = "Model training for k-nearest neighbors")


