library(caret)
dataset = read.table("C:/Users/Administrator/Desktop/PB/datasets/adult.csv", sep=",", header=F, fill=FALSE, strip.white=T)
#data(iris)
# rename the dataset
#dataset <- iris
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$V15, p=0.80, list=FALSE)
# select 20% of the data for validation☼
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

write.csv(dataset, file = "C:/Users/Administrator/Desktop/PB/datasets/adult_train.csv")
write.csv(validation, file = "C:/Users/Administrator/Desktop/PB/datasets/adult_test.csv")