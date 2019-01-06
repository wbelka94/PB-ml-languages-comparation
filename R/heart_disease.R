library(caret)
library(tictoc)
# attach the iris dataset to the environment
heart <- read.csv('C:/Users/Administrator/Desktop/R-heart_disease/data.csv')
# rename the dataset
dataset <- heart
print(heart)
dataset$num <- ifelse(dataset$num == 0, 'true', 'false')

print(dataset)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$num, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]


# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
tic.clearlog()
set.seed(7)
tic("lda")
fit.lda <- train(num~., data=dataset, method="lda", metric=metric, trControl=control)
toc(log = TRUE, quiet = TRUE)

# b) nonlinear algorithms
# CART
set.seed(7)
tic("cart")
fit.cart <- train(num~., data=dataset, method="rpart", metric=metric, trControl=control)
toc(log = TRUE, quiet = TRUE)
# kNN
set.seed(7)
tic("knn")
fit.knn <- train(num~., data=dataset, method="knn", metric=metric, trControl=control)
toc(log = TRUE, quiet = TRUE)
# c) advanced algorithms
# SVM
set.seed(7)
tic("svm")
fit.svm <- train(num~., data=dataset, method="svmRadial", metric=metric, trControl=control)
toc(log = TRUE, quiet = TRUE)
# Random Forest
set.seed(7)
tic("rf")
fit.rf <- train(num~., data=dataset, method="rf", metric=metric, trControl=control)
toc(log = TRUE, quiet = TRUE)
# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$num)
timings <- tic.log()
print(timings)