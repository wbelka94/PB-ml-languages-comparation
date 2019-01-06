# The Rdataset shall be added, if unavailable.
using CSV

function convertQuality(y)
    for r = 1:size(y)[1]
        if (y[r] >= 5)
            y[r] = 1
        else
            y[r] = 0
        end
    end
    return y
end

train =  CSV.read("C:/Users/Administrator/Desktop/PB/datasets/wine_train.csv" ; delim=',', header=true)
test =  CSV.read("C:/Users/Administrator/Desktop/PB/datasets/wine_test.csv" ; delim=',', header=true)

# print(first(train,5))
# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
train_X = convert(Array, train[:,1:13])
train_y = convert(Array, train[:,14])
train_X = float.(train_X)
train_y = convertQuality(train_y)
test_X = convert(Array, test[:,1:13])
test_y = convert(Array, test[:,14])
test_X = float.(test_X)
test_y = convertQuality(test_y)
# Load the Logistic Regression model
using ScikitLearn
# This model requires scikit-learn. See
# http://scikitlearnjl.readthedocs.io/en/latest/models/#installation
@sk_import linear_model: LogisticRegression
@sk_import neighbors: KNeighborsClassifier
# The Hyperparameters such as regression strength, whether to fit the intercept, penalty type.
model = KNeighborsClassifier()
# model = LogisticRegression()

# Train the model.
fit!(model, train_X, train_y)
# Accuracy is evaluated
accuracy = sum(predict(model, test_X) .== test_y) / length(test_y)
println("accuracy: $accuracy")
