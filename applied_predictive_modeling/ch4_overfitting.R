##############################
# Ch. 4 - Over-Fitting
# Following along
#
# Josh Jensen
##############################


library(AppliedPredictiveModeling)
library(tidyverse)
library(caret)


#
### data splitting
#

data(twoClassData)

str(predictors)
str(classes)

set.seed(1)

trainingRows <- createDataPartition(classes,
                                p = .80,
                                list = F)
head(trainingRows)

trainPredictors <- predictors[trainingRows,]
testPredictors <- predictors[-trainingRows,]

trainClasses <- classes[trainingRows]
testClasses <- classes[-trainingRows]

str(trainPredictors)
str(testPredictors)



#
### resampling
#

set.seed(1)

# repeated samples
repeatedSplits <- createDataPartition(trainClasses,
                                      p = .8,
                                      times = 3)
str(repeatedSplits)

# k folds
cvSplits <- createFolds(trainClasses,
                        k = 10,
                        returnTrain = T)
str(cvSplits)


#
### model building
#

trainPredictors <- as.matrix(trainPredictors)
testPredictors <- as.matrix(testPredictors)
trainClasses <- as.factor(trainClasses)


knnFit <- caret::knn3(x = trainPredictors,
               y = trainClasses,
               k = 5)
knnFit


testPredictions <- predict(knnFit, 
                           newdata = testPredictors,
                           type = "class")
head(testPredictions)
str(testPredictions)



#
### hyperparameter tuning 
#

data(GermanCredit)

train_split <- createDataPartition(GermanCredit$Class,
                                   p = .8,
                                   list = F)
train_GermanCredit <- GermanCredit[train_split,]
test_GermanCredit <- GermanCredit[-train_split,]

set.seed(1056)
options(warn=-1)

svmFit <- train(Class ~ .,
                data = train_GermanCredit,
                method = 'svmRadial')
svmFit

svmFit2 <- train(Class ~ .,
                 data = train_GermanCredit,
                 method = 'svmRadial',
                 preProc = c('center','scale'))
svmFit2

svmFit3 <- train(Class ~ .,
                 data = train_GermanCredit,
                 method = 'svmRadial',
                 preProc = c('center', 'scale'),
                 tuneLength = 10)
svmFit3

set.seed(1056)
svmFit4 <- train(Class ~ .,
                 data = train_GermanCredit,
                 method = 'svmRadial',
                 preProc = c('center', 'scale'),
                 tuneLength = 10,
                 trControl = trainControl(method = 'repeatedcv',
                                          repeats = 5,
                                          classProbs = T))
svmFit4

plot(svmFit4, scales = list(x = list(log = 2)))

predictedProbs <- predict(svmFit4,
                          newdata = test_GermanCredit,
                          type = 'prob')
head(predictedProbs)



#
### between model comps
#

set.seed(1056)
logFit <- train(Class ~ .,
                data = train_GermanCredit,
                method = 'glm',
                trControl = trainControl(method = 'repeatedcv',
                                         repeats = 5))
logFit

resamp <- resamples(list(SVM = svmFit4,
                         Logistic = logFit))
summary(resamp)

modelDiffs <- diff(resamp)
summary(modelDiffs)
