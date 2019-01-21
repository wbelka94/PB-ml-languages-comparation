import pandas as pd
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import psutil
import os
from timeit import default_timer as timer

from sklearn.metrics import accuracy_score

from convertFeaturesToFloat import DatasetConverter

train = pd.read_csv('../datasets/iris_train.csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('../datasets/iris_test.csv')
test = test.drop(test.columns[0], axis=1)
dc = DatasetConverter()
dc.convertAndSave(train, test, 'iris', '../datasets/')

train = pd.read_csv('../datasets/iris_train[float].csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('../datasets/iris_test[float].csv')
test = test.drop(test.columns[0], axis=1)
train_Y = train.iloc[:, -1]
train_X = train.iloc[:, :-1]
test_Y = test.iloc[:, -1]
test_X = test.iloc[:, :-1]

for i in range(10):
    clf = neighbors.KNeighborsClassifier()
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("Fit time")
    print(end-start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("Pred time")
    print(end-start)

    clf = tree.DecisionTreeClassifier()
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("Fit time")
    print(end-start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("Pred time")
    print(end-start)

    clf = RandomForestClassifier()
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("Fit time")
    print(end-start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("Pred time")
    print(end-start)

    clf = svm.SVC(gamma='scale')
    # #
    start = timer()
    clf.fit(train_X, train_Y)
    end = timer()
    print("Fit time")
    print(end-start)
    #
    start = timer()
    predicted_y = clf.predict(test_X)
    end = timer()
    print("Pred time")
    print(end-start)

    print(accuracy_score(test_Y, predicted_y))
    temp = dict(psutil.virtual_memory()._asdict())
    print(temp)
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse)