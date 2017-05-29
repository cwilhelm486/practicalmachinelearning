# Prediction of Exercise Manner
Caitlin  
May 21, 2017  



##Summary
This project was done to predict the manner in which participants did the weight lifting exercise given, listed as the `classe` variable in the given dataset. Data was collected from accelerometers on the belt, forearm, arm, and dumbbell of each participant. There are five classes of the outcome variable, with 159 predictors.

##Preparation
To begin with, necessary packages and data are loaded.


```r
library(caret)
library(randomForest)

training <- read.csv("C:/Users/Caitlin/Documents/Coursera/pml-training.csv")
testing <- read.csv("C:/Users/Caitlin/Documents/Coursera/pml-testing.csv")
```

In order to have the best environment for model building, the variables with near-zero variance or NA values are removed from both the testing and training sets, as well as categorical variables such as `user_name` that will have no effect on the model. If necessary, variables can be added back in with missing values imputed.


```r
nsv <- nearZeroVar(training,saveMetrics=TRUE)
training_no_nzv <- training[,!nsv$nzv]
testing_no_nzv <- testing[,!nsv$nzv]

training_no_na <- training_no_nzv[,(colSums(is.na(training_no_nzv)) == 0)]
testing_no_na <- testing_no_nzv[,(colSums(is.na(testing_no_nzv)) == 0)]

remove_col_train <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window")
remove_col_test <- c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","num_window","problem_id")
training_small <- training_no_na[,!(names(training_no_na) %in% remove_col_train)]
testing_small <- testing_no_na[,!(names(testing_no_na) %in% remove_col_test)]
```

In order to do cross-validation, the training set is split into training and validation sets.


```r
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_final <- training_small[inTrain,]
validation <- training_small[-inTrain,]
```

##Model Building
A random forest model seems like the best fit for the data. After building the model, predict on the validation set and check the confusion matrix.


```r
set.seed(333)
modFit1 <- train(classe~., method = 'rf', data = training_final, trControl = trainControl(method = 'cv', number = 4))

pred <- predict(modFit1, newdata = validation)
confusionMatrix(pred, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    6    0    0    0
##          B    1 1133    6    0    0
##          C    0    0 1018   11    0
##          D    0    0    2  953    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9947   0.9922   0.9886   0.9963
## Specificity            0.9986   0.9985   0.9977   0.9988   1.0000
## Pos Pred Value         0.9964   0.9939   0.9893   0.9937   1.0000
## Neg Pred Value         0.9998   0.9987   0.9984   0.9978   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1925   0.1730   0.1619   0.1832
## Detection Prevalence   0.2853   0.1937   0.1749   0.1630   0.1832
## Balanced Accuracy      0.9990   0.9966   0.9950   0.9937   0.9982
```

```r
modFit1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.66%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    2    0    0    0 0.0005120328
## B   14 2639    5    0    0 0.0071482318
## C    0   16 2376    4    0 0.0083472454
## D    0    0   40 2210    2 0.0186500888
## E    0    0    2    6 2517 0.0031683168
```

```r
accuracy <- sum(pred == validation$classe) / length(pred)

error <- (1 - accuracy) * 100
```

The random forest model has an accuracy of 0.9949023, so the out-of-sample error estimate is 0.51%. These are both indicators of good model fit, so there is no need to redo the model with missing values imputed.

##Prediction
Finally the model is used to predict on the test set.


```r
test_pred <- predict(modFit1, testing_small)

final <- data.frame(Predictions = character(), stringsAsFactors = FALSE)

for (i in 1:length(test_pred)) {
  final[i, 1] <- paste("problem_id ", i, ": ", test_pred[i])
}

final
```

```
##            Predictions
## 1   problem_id  1 :  B
## 2   problem_id  2 :  A
## 3   problem_id  3 :  B
## 4   problem_id  4 :  A
## 5   problem_id  5 :  A
## 6   problem_id  6 :  E
## 7   problem_id  7 :  D
## 8   problem_id  8 :  B
## 9   problem_id  9 :  A
## 10 problem_id  10 :  A
## 11 problem_id  11 :  B
## 12 problem_id  12 :  C
## 13 problem_id  13 :  B
## 14 problem_id  14 :  A
## 15 problem_id  15 :  E
## 16 problem_id  16 :  E
## 17 problem_id  17 :  A
## 18 problem_id  18 :  B
## 19 problem_id  19 :  B
## 20 problem_id  20 :  B
```




