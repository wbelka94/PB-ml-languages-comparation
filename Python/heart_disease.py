import pandas as pd
from sklearn import neighbors, tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer


def convertNum(dataset, numColumn):
    for r, row in dataset.iterrows():
        if (row[numColumn] > 0):
            dataset.set_value(r, numColumn, 1)
        else:
            dataset.set_value(r, numColumn, 0)
    return dataset


train = pd.read_csv('C:/Users/Administrator/Desktop/PB/datasets/heart_disease_train.csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('C:/Users/Administrator/Desktop/PB/datasets/heart_disease_test.csv')
test = test.drop(test.columns[0], axis=1)
train = convertNum(train, "num")
test = convertNum(test, "num")
print(train)

train_Y = train.iloc[:, -1]
train_X = train.iloc[:, :-1]
test_Y = test.iloc[:, -1]
test_X = test.iloc[:, :-1]

for i in range(10):
    clf = neighbors.KNeighborsClassifier()
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("knn-fit: ", end - start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("knn-predict: ", end - start)
    print("knn-accuracy: ", accuracy_score(test_Y, predicted_y))

    clf = tree.DecisionTreeClassifier()
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("tree-fit: ", end - start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("tree-predict: ", end - start)
    print("tree-accuracy: ", accuracy_score(test_Y, predicted_y))

    clf = RandomForestClassifier()
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("RandomForest-fit: ", end - start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("RandomForest-predict: ", end - start)
    print("RandomForest-accuracy: ", accuracy_score(test_Y, predicted_y))

    clf = svm.SVC(gamma='scale')
    # #
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("svc-fit: ", end - start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("svc-predict: ", end - start)
    print("svc-accuracy: ", accuracy_score(test_Y, predicted_y))