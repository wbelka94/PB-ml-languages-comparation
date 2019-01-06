import pandas as pd
from pandas import DataFrame


class DatasetConverter:
    def __init__(self):
        self.featuresValues = {}

    def convertFeaturesToFloat(self, dataset):
        newDataset = {}
        for r, row in dataset.iterrows():
            newDataset[r] = []
            for c, value in enumerate(row):
                featureValues = self.featuresValues.get(c, [])
                try:
                    try:
                        newValue = float(value)
                    except ValueError:
                        newValue = featureValues.index(value)
                except ValueError:
                    featureValues.append(value)
                    newValue = featureValues.index(value)
                    self.featuresValues[c] = featureValues
                newDataset[r].insert(c, newValue)
        return pd.DataFrame.from_dict(newDataset, orient="index")

    def convertAndSave(self, train, test, basename, dir):
        train = self.convertFeaturesToFloat(train)
        test = self.convertFeaturesToFloat(test)
        train.to_csv(dir + basename + "_train[float].csv", ',')
        test.to_csv(dir + basename + "_test[float].csv", ',')