import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score


def convertQuality(dataset, qualityColumn):
    for r, row in dataset.iterrows():
        if (row[qualityColumn] >= 5):
            dataset.set_value(r, qualityColumn, 1)
        else:
            dataset.set_value(r, qualityColumn, 0)
    return dataset


train = pd.read_csv('../datasets/wine_train.csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('../datasets/wine_test.csv')
test = test.drop(test.columns[0], axis=1)
train = convertQuality(train, "quality")
test = convertQuality(test, "quality")

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
