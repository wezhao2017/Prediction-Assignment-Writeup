---
title: "Practical Machine Learning Project: Prediction Assignment Writeup"
author: "wezhao2017"
date: "1-30-2018"
output: 
  html_document: 
    keep_md: yes
---



## Introduction

This project is a Practical Machine Learning course assignment. The goal is to predict the manner in which they practice the exercise and set the variable of classification in the training. We will use the data from accelerometers of six participants such as the belt, forearm, arm, and dumbell. This report describes how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we did. We will also use our prediction model to predict 20 different test cases. More information is avaliable from the website at: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.


## Loading the data

loading the packages needed and getting the training data and testing data from the URLs given.


```r
library(lattice)
library(ggplot2)
library(caret)
library(gbm)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.3
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(11111)
TrainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(TrainUrl,na.strings = c("NA", ""))
testing <- read.csv(TestUrl,na.strings = c("NA", ""))
```

## Cleaning the data

Deleting columns of the raw data that contain any missing values, and removing the invalid variables.


```r
training <- training [, colSums(is.na(training))==0]
testing <- testing [, colSums(is.na(testing))==0]

classe <-training$classe
trainingremove <- grepl("^X|timestamp|window",names(training))
training <- training [, !trainingremove]
trainclean <- training [, sapply(training, is.numeric)]
trainclean$classe <- classe

testingremove <- grepl("^X|timestamp|window",names(testing))
testing <- testing [, !testingremove]
testclean <- testing [, sapply(testing, is.numeric)]
```

## Splitting the data

Seleting the train data(mytrain, 60%) from cleaned data, and a validation set (mytest) to test the out-of-sample errors.


```r
inTrain <- createDataPartition(trainclean$classe, p= .6, list = FALSE, times = 1)
mytrain <- trainclean[inTrain,]
mytest <- trainclean[-inTrain,]

dim(mytrain)
```

```
## [1] 11776    53
```

```r
dim(mytest)
```

```
## [1] 7846   53
```

## Prediction with Grandient Boosting Method

Using grandient boosting method to predict the outcome.


```r
set.seed(11111)
controlgbm<- trainControl(method = "cv", 5)
modelgbm <- train(classe~., data=mytrain, method = "gbm", trControl = controlgbm, verbose = FALSE)
modelgbm
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9423, 9420, 9420, 9420, 9421 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7507637  0.6840971
##   1                  100      0.8210744  0.7735067
##   1                  150      0.8514767  0.8120988
##   2                   50      0.8485036  0.8080900
##   2                  100      0.9037872  0.8782391
##   2                  150      0.9307904  0.9124058
##   3                   50      0.8912182  0.8622356
##   3                  100      0.9392824  0.9231572
##   3                  150      0.9591538  0.9483223
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
plot(modelgbm)
```

![](Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
predictiongbm <- predict(modelgbm, mytest)
confusionMatrix(predictiongbm,mytest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2210   54    0    0    1
##          B   14 1417   29    9   25
##          C    6   42 1315   34    9
##          D    2    4   21 1235   24
##          E    0    1    3    8 1383
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9635          
##                  95% CI : (0.9592, 0.9676)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9539          
##  Mcnemar's Test P-Value : 1.134e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9901   0.9335   0.9613   0.9603   0.9591
## Specificity            0.9902   0.9878   0.9860   0.9922   0.9981
## Pos Pred Value         0.9757   0.9485   0.9353   0.9603   0.9914
## Neg Pred Value         0.9961   0.9841   0.9918   0.9922   0.9909
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2817   0.1806   0.1676   0.1574   0.1763
## Detection Prevalence   0.2887   0.1904   0.1792   0.1639   0.1778
## Balanced Accuracy      0.9902   0.9606   0.9736   0.9763   0.9786
```

```r
errorout <- 1 - as.numeric(confusionMatrix(predictiongbm,mytest$classe)$overall[1])
errorout
```

```
## [1] 0.0364517
```

The estimated accuracy of model is 96.35%, and the estimated out-of-sample error is 3.65%.  


## Prediction with Random Forests

Using random forests to predict the outcome.


```r
set.seed(11111)
controlrF<- trainControl(method = "cv", 5)
modelrF <- train(classe~., data=mytrain, method = "rf", trControl = controlrF, ntree = 250)
modelrF
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9423, 9420, 9420, 9420, 9421 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9883661  0.9852822
##   27    0.9890456  0.9861420
##   52    0.9842890  0.9801251
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

```r
predictionrF <- predict(modelrF, mytest)
confusionMatrix(predictionrF,mytest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2226   18    0    0    0
##          B    4 1494    4    2    3
##          C    2    6 1355   19    2
##          D    0    0    9 1263    6
##          E    0    0    0    2 1431
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9902          
##                  95% CI : (0.9877, 0.9922)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9876          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9973   0.9842   0.9905   0.9821   0.9924
## Specificity            0.9968   0.9979   0.9955   0.9977   0.9997
## Pos Pred Value         0.9920   0.9914   0.9790   0.9883   0.9986
## Neg Pred Value         0.9989   0.9962   0.9980   0.9965   0.9983
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2837   0.1904   0.1727   0.1610   0.1824
## Detection Prevalence   0.2860   0.1921   0.1764   0.1629   0.1826
## Balanced Accuracy      0.9971   0.9911   0.9930   0.9899   0.9960
```

```r
errorout <- 1 - as.numeric(confusionMatrix(predictionrF,mytest$classe)$overall[1])
errorout
```

```
## [1] 0.009813918
```

The estimated accuracy of model is 99.02%, and the estimated out-of-sample error is 0.98%.  

## Predicting on test set

From the above datasets shows that the random forest model is better than the other one. We choose it to predict the values for the test data set.


```r
outcome <- predict(modelrF, testclean [, -length(names(testclean))])
outcome
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```























