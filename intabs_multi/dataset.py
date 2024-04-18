import pandas as pd
import numpy as np
from carla import Data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing


# interfacing carla dataset classes
class InnDataSet(Data):
    def __init__(self, name):
        df = None
        continuous = None
        if name == "iris":
            data_iris = load_iris()
            X_val = min_max_scale(data_iris.data)
            y_val = data_iris.target.reshape(-1, 1)
            dataset_val = np.concatenate((X_val, y_val), axis=1)
            np.random.seed(20)
            np.random.shuffle(dataset_val)
            df = pd.DataFrame(data=dataset_val, columns=["feature1", "feature2", "feature3", "feature4", "target"])
            continuous = ["feature1", "feature2", "feature3", "feature4"]
        if name == "housing":
            housing = fetch_california_housing()
            X_val = min_max_scale(housing.data)
            y_val = housing.target.reshape(-1, 1)
            y_val_regression = y_val.flatten()
            lower_cut = np.percentile(y_val_regression, 33)
            higher_cut = np.percentile(y_val_regression, 67)
            class0_idx = np.where(y_val_regression <= lower_cut)[0]
            class1_idx = np.intersect1d(np.where(y_val_regression > lower_cut)[0],
                                        np.where(y_val_regression <= higher_cut)[0])
            class2_idx = np.where(y_val_regression > higher_cut)[0]
            y_val_regression[class0_idx] = 0
            y_val_regression[class1_idx] = 1
            y_val_regression[class2_idx] = 2
            y_val = y_val_regression.reshape(-1, 1)
            dataset_val = np.concatenate((X_val, y_val), axis=1)
            np.random.seed(20)
            np.random.shuffle(dataset_val)
            df = pd.DataFrame(data=dataset_val,
                              columns=["feature1", "feature2", "feature3", "feature4", "feature5", "feature6",
                                       "feature7", "feature8", "target"])
            continuous = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6",
                          "feature7", "feature8", ]
        self.pcontinuous = continuous
        self.pdf = df
        self.ptarget = "target"
        target = "target"

        # put target column to the last
        dfx, dfy = self.pdf.drop(columns=[target]), pd.DataFrame(self.pdf[target])
        self.pdf = pd.concat([dfx, dfy], axis=1)

        self.pX, self.py = self.pdf.drop(columns=[target]), pd.DataFrame(self.pdf[target])

        # divide to df1 and df2 for following retraining
        self.pfeat_var_map = {}
        for i in range(len(self.pX.columns)):
            self.pfeat_var_map[i] = [i]
        size = self.pdf.shape[0]
        np.random.seed(1)
        idx_1 = np.sort(np.random.choice(size, int(size / 2) - 1, replace=False))
        idx_2 = np.array([i for i in np.arange(size) if i not in idx_1])
        self.pX1 = pd.DataFrame(data=self.pX.values[idx_1], columns=self.pX.columns)
        self.py1 = pd.DataFrame(data=self.py.values[idx_1], columns=self.py.columns)
        self.pX2 = pd.DataFrame(data=self.pX.values[idx_2], columns=self.pX.columns)
        self.py2 = pd.DataFrame(data=self.py.values[idx_2], columns=self.py.columns)
        self.pX1_train, self.pX1_test, self.py1_train, self.py1_test = train_test_split(self.pX1, self.py1,
                                                                                        stratify=self.py1,
                                                                                        test_size=0.2, shuffle=True,
                                                                                        random_state=0)
        self.pX2_train, self.pX2_test, self.py2_train, self.py2_test = train_test_split(self.pX2, self.py2,
                                                                                        stratify=self.py2,
                                                                                        test_size=0.2, shuffle=True,
                                                                                        random_state=0)

    @property
    def columns(self):
        return list(self.pdf.columns)

    @property
    def features(self):
        return [i for i in list(self.pdf.columns) if i != self.target]

    @property
    def categorical(self):
        return [item for item in self.pX.columns if item not in self.pcontinuous]

    @property
    def continuous(self):
        return self.pcontinuous

    @property
    def immutables(self):
        return []

    @property
    def target(self):
        return self.ptarget

    @property
    def df(self):
        return self.pdf

    @property
    def df_train(self):
        return self.pX1_train

    @property
    def df_test(self):
        return self.pX1_test

    @property
    def X1(self):
        return self.pX1

    @property
    def y1(self):
        return self.py1

    @property
    def X2(self):
        return self.pX2

    @property
    def y2(self):
        return self.py2

    @property
    def X1_train(self):
        return self.pX1_train

    @property
    def X1_test(self):
        return self.pX1_test

    @property
    def y1_test(self):
        return self.py1_test

    @property
    def y2_train(self):
        return self.py2_train

    @property
    def X2_train(self):
        return self.pX2_train

    @property
    def X2_test(self):
        return self.pX2_test

    @property
    def y2_test(self):
        return self.py2_test

    @property
    def y1_train(self):
        return self.py1_train

    @property
    def ordinal_features(self):
        return {}

    @property
    def continuous_features(self):
        return self.pcontinuous

    @property
    def discrete_features(self):
        lst = [item for item in self.pX.columns if item not in self.pcontinuous]
        # squash binary OHEs to 1
        disc = {}
        for item in lst:
            disc[item] = 1
        return disc

    @property
    def feat_var_map(self):
        return self.pfeat_var_map

    def transform(self, df):
        return df

    def inverse_transform(self, df):
        return df


def min_max_scale(data_x):
    return (data_x - np.min(data_x, axis=0)) / (np.max(data_x, axis=0) - np.min(data_x, axis=0))
