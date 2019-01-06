import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score

from utils.convertFeaturesToFloat import DatasetConverter

train = pd.read_csv('../datasets/adult_train.csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('../datasets/adult_test.csv')
test = test.drop(test.columns[0], axis=1)
dc = DatasetConverter()
dc.convertAndSave(train, test, 'adult', '../datasets/')

train = pd.read_csv('../datasets/adult_train[float].csv')
train = train.drop(train.columns[0], axis=1)
test = pd.read_csv('../datasets/adult_test[float].csv')
test = test.drop(test.columns[0], axis=1)

print(test)

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
