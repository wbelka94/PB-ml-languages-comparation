# The Rdataset shall be added, if unavailable.
using CSV
using IJulia
IJulia.installkernel("Julia nodeps", "--depwarn=no")

function convertNum(y)
    for r = 1:size(y)[1]
        if (y[r] == 0)
            y[r] = 0
        else
            y[r] = 1
        end
    end
    return y
end

train =  CSV.read("C:/Users/Administrator/Desktop/PB/datasets/heart_disease_train.csv" ; delim=',', header=true)
test =  CSV.read("C:/Users/Administrator/Desktop/PB/datasets/heart_disease_test.csv" ; delim=',', header=true)
# print(first(train,5))
# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
train_X = convert(Array, train[:,2:14])
train_y = convert(Array, train[:,15])
train_X = float.(train_X)
train_y = convertNum(train_y)
test_X = convert(Array, test[:,2:14])
test_y = convert(Array, test[:,15])
test_X = float.(test_X)
test_y = convertNum(test_y)
# Load the Logistic Regression model
using ScikitLearn
# This model requires scikit-learn. See
# http://scikitlearnjl.readthedocs.io/en/latest/models/#installation
@sk_import linear_model: LogisticRegression
@sk_import neighbors: KNeighborsClassifier
# The Hyperparameters such as regression strength, whether to fit the intercept, penalty type.
@sk_import neighbors: KNeighborsClassifier
@sk_import tree: DecisionTreeClassifier
@sk_import ensemble: RandomForestClassifier
@sk_import svm: SVC

for i = 1:10
    model = KNeighborsClassifier()

    startTime = time_ns()
    fit!(model, train_X, train_y)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("knn-fit: $elapsed")
    startTime = time_ns()
    prediction = predict(model, test_X)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("knn-predict: $elapsed")
    accuracy = sum(prediction .== test_y) / length(test_y)
    println("knn-accuracy: $accuracy")

    model = DecisionTreeClassifier()

    startTime = time_ns()
    fit!(model, train_X, train_y)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("tree-fit: $elapsed")
    startTime = time_ns()
    prediction = predict(model, test_X)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("tree-predict: $elapsed")
    accuracy = sum(prediction .== test_y) / length(test_y)
    println("tree-accuracy: $accuracy")

    model = RandomForestClassifier()

    startTime = time_ns()
    fit!(model, train_X, train_y)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("RandomForest-fit: $elapsed")
    startTime = time_ns()
    prediction = predict(model, test_X)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("RandomForest-predict: $elapsed")
    accuracy = sum(prediction .== test_y) / length(test_y)
    println("RandomForest-accuracy: $accuracy")

    model = SVC()

    startTime = time_ns()
    fit!(model, train_X, train_y)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("SVC-fit: $elapsed")
    startTime = time_ns()
    prediction = predict(model, test_X)
    endTime = time_ns()
    elapsed = endTime - startTime
    println("SVC-predict: $elapsed")
    accuracy = sum(prediction .== test_y) / length(test_y)
    println("SVC-accuracy: $accuracy")
end
