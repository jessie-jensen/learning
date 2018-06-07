##############################
# Ch. 5 - Model Performance & Bias-Variance Tradeoff
# Following along w/ compute section
#
# Josh Jensen
##############################


library(AppliedPredictiveModeling)
library(tidyverse)
library(caret)

observed <- c(.22, .83, -.12, .89, -.23, -1.3, -.15, -1.4, .62, .99, -.18, .32, .34, -.3, .04, -.87, .55, -1.3, -1.15, .2)
predicted <- c(.24, .78, -.66, .53, .7, -.75, -.41, -.43, .49, .79, -1.19, .06, .75, -.07, .43, -.42, -.25, -.64, -1.26, -.07)

residualValues <- observed - predicted
summary(residualValues)

# plots
axisRange <- extendrange(c(observed, predicted))

plot(predicted, observed,
     ylim = axisRange,
     xlim = axisRange)
abline(0, 1, col='darkgrey', lty = 2)


plot(predicted, residualValues,
     ylab = 'residual')
abline(h = 0, col='darkgrey', lty=2)

# r2 & rmse
R2(predicted, observed)
RMSE(predicted, observed)

# correlations
cor(predicted, observed) #simple
cor(predicted, observed, method = 'spearman') #rank
