import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score


def convertNum(dataset, numColumn):
    for r, row in dataset.iterrows():
        if (row[numColumn] > 0):
            dataset.set_value(r, numColumn, 1)
        else:
            dataset.set_value(r, numColumn, 0)
    return dataset


train = pd.read_csv('../datasets/heart_disease_train.csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('../datasets/heart_disease_test.csv')
test = test.drop(test.columns[0], axis=1)
train = convertNum(train, "num")
test = convertNum(test, "num")
print(train)

train_Y = train.iloc[:, -1]
train_X = train.iloc[:, :-1]
test_Y = test.iloc[:, -1]
test_X = test.iloc[:, :-1]

clf = neighbors.KNeighborsClassifier()
# #
clf.fit(train_X, train_Y)
#
predicted_y = clf.predict(test_X)
print(accuracy_score(test_Y, predicted_y))
