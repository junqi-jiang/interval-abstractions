import pandas as pd
import numpy as np
from carla import Data
from carla.data.catalog import OnlineCatalog
from sklearn.model_selection import train_test_split


# interfacing carla dataset classes
class InnDataSet(Data):
    def __init__(self, name):
        self.online_dataset = OnlineCatalog(name)
        self.pdf = self.online_dataset.df
        target = self.online_dataset.target

        if name == "give_me_some_credit":
            new_df_0_vals = self.pdf[self.pdf[target] == 0].values
            new_df = pd.DataFrame(data=np.concatenate((self.pdf.values, new_df_0_vals), axis=0), columns=self.pdf.columns)
            self.pdf = new_df
            self.pdf = pd.DataFrame(data=np.concatenate((new_df_0_vals, self.pdf.values), axis=0), columns=self.pdf.columns)
            self.pdf = self.pdf.sample(frac=1)

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
        return [i for i in list(self.pdf.columns) if i != self.online_dataset.target]

    @property
    def categorical(self):
        return [item for item in self.pX.columns if item not in self.online_dataset.continuous]

    @property
    def continuous(self):
        return self.online_dataset.continuous

    @property
    def immutables(self):
        imm = []
        for item in self.online_dataset.immutables:
            for name in self.pdf.columns:
                if item in name:
                    imm.append(name)
        if len(self.online_dataset.immutables) == 0:
            imm.append(self.continuous[-1])  # ensure we can run FACE... Couldn't resolve the bugs in carla
        return imm

    @property
    def target(self):
        return self.online_dataset.target

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
        return self.online_dataset.continuous

    @property
    def discrete_features(self):
        lst = [item for item in self.pX.columns if item not in self.online_dataset.continuous]
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
