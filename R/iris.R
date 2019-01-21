library(caret)
library(tictoc)
# attach the iris dataset to the environment
#data(iris)
# rename the dataset
# create a list of 80% of the rows in the original dataset we can use for training
#validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
#validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
#dataset <- dataset[validation_index,]

validation <- read.table("C:/Users/Administrator/Desktop/PB/datasets/iris_test.csv", sep=",", header=T, fill=FALSE, strip.white=T)
dataset <- read.table("C:/Users/Administrator/Desktop/PB/datasets/iris_train.csv", sep=",", header=T, fill=FALSE, strip.white=T)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=2)
metric <- "Accuracy"
for (i in 1:10){
  # a) linear algorithms
  tic.clearlog()
  set.seed(7)
  tic("lda-fit")
  fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
  toc(log = TRUE, quiet = TRUE)
  
  # b) nonlinear algorithms
  # CART
  set.seed(7)
  tic("cart-fit")
  fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
  toc(log = TRUE, quiet = TRUE)
  # kNN
  set.seed(7)
  tic("knn-fit")
  fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
  toc(log = TRUE, quiet = TRUE)
  # c) advanced algorithms
  # SVM
  set.seed(7)
  tic("svm-fit")
  fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
  toc(log = TRUE, quiet = TRUE)
  # Random Forest
  set.seed(7)
  tic("rf-fit")
  fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)
  toc(log = TRUE, quiet = TRUE)
  # summarize accuracy of models
  results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
  summary(results)
  
  tic("lda-predict")
  x = predict(fit.lda, validation)
  toc(log = TRUE, quiet = TRUE)
  tic("cart-predict")
  x = predict(fit.cart, validation)
  toc(log = TRUE, quiet = TRUE)
  tic("knn-predict")
  x = predict(fit.knn, validation)
  toc(log = TRUE, quiet = TRUE)
  tic("svm-predict")
  x = predict(fit.svm, validation)
  toc(log = TRUE, quiet = TRUE)
  tic("rf-predict")
  x = predict(fit.rf, validation)
  toc(log = TRUE, quiet = TRUE)
  
  # estimate skill of LDA on the validation dataset
  # predictions <- predict(fit.svm, validation)
  # confusionMatrix(predictions, validation$Species)
  timings <- tic.log()
  print(timings)
}
