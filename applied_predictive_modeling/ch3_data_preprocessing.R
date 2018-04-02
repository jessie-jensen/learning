##############################
# Ch. 3 - Data Preprocessing
# Following along
#
# Josh Jensen
##############################

library(tidyverse)
library(magrittr)
library(AppliedPredictiveModeling)
library(e1071)
library(caret)
library(corrplot)

# load data
data("segmentationOriginal")
segData <- subset(segmentationOriginal, Case == 'Train')

# save id & features from the original experiment as outcome as vectors & 
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case

# remove unwanted columns
segData %<>% select(-one_of('Cell', 'Class', 'Case'))
segData %<>% select(-contains('Status'))

# check remaining types of features
table(sapply(segData, class))



#
### boxcox transform
#

# get all skew values
skewValues <- sapply(segData, skewness)
head(skewValues)

# check skewness on one feature, then boxcox transform
skewness(segData$AreaCh1)
x <- BoxCoxTrans(segData$AreaCh1)
x

head(segData$AreaCh1)
predict(x, head(segData$AreaCh1))



#
### apply pca
#

pca <- prcomp(segData,
              center = T,
              scale. = T)

head(pca$x[,1:5])

head(pca$rotation[,1:5])

# calc cumulative percent of variance of each pricipal component
pct_var <- (pca$sdev^2 / sum(pca$sdev^2)) * 100
head(pct_var)


# use caret to put all together
trans <- preProcess(segData,
                    method = c('BoxCox', 'center', 'scale', 'pca'))
trans

transformed <- predict(trans, segData)



#
### filtering
#

# check for any features with near 0 variance
nearZeroVar(segData)

corrs <- cor(segData)
dim(corrs)

corrplot(corrs, orders ='hclust')

# filter on high correlation / collinearity
highCorr <- findCorrelation(corrs, cutoff = .75)
highCorr

filtered_seg <- segData[, -highCorr]



#
### creating dummy variables
#

data("cars")

cars %<>% select(one_of('Price', 'Mileage', 'Cylinder'))
cars$Cylinder <- as.factor(cars$Cylinder)

head(cars)


simpleMod <- dummyVars(~ Mileage + Cylinder,
                       data = cars,
                       levelsOnly = T)
simpleMod
predict(simpleMod, head(cars))

# now with interaction term
modWithInteraction <- dummyVars(~ Mileage + Cylinder + Mileage:Cylinder,
                                data = cars,
                                levelsOnly = T)
modWithInteraction
predict(modWithInteraction, head(cars))
