import copy

from carla import MLModel
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import sys
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# TORCH TRAINING DATASET UTIL CLASS
class TrainingDataset(Dataset):
    def __init__(self, X, y):
        try:
            self.X = X.values
            self.y = y.values
        except:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.y[idx]), float()



def train_clf(X1_train, y1_train, X1_test, y1_test, num_hidden_neurons=5, epochs=50, data_name="adult", save_clf=False,
              load_clf=False, lr=0.01, eval=True):
    if load_clf:
        model = torch.load(f"./{data_name}.pth")
        # eval
        print("Evaluations on training data")
        res = model(torch.tensor(X1_train.values).float()).detach().numpy().flatten().round()
        print('\n',
              classification_report(y1_train, res, target_names=[f'bad credit (0)', f'good credit (1)'], digits=3))
        print("Evaluations on testing data")
        res = model(torch.tensor(X1_test.values).float()).detach().numpy().flatten().round()
        print('\n', classification_report(y1_test, res, target_names=[f'bad credit (0)', f'good credit (1)'], digits=3))
        return model

    trainds = TrainingDataset(X1_train, y1_train)
    testds = TrainingDataset(X1_test, y1_test)
    params = {'batch_size': 32,
              'shuffle': True}
    traindl = DataLoader(trainds, **params)
    testdl = DataLoader(testds, **params)

    input_size = X1_train.values.shape[1]

    model = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_neurons, num_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_neurons, 3),
            torch.nn.Softmax(dim=1)
        )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l2lamb = 0.0001
    for ep in range(epochs):
        for (Xb, yb, _) in traindl:
            model.train()
            optimizer.zero_grad()
            y1_pred = model(torch.tensor(Xb).float())
            loss_pred = loss_fn(y1_pred, torch.tensor(yb).long().view(-1))
            l2_norm = sum(val.pow(2.0).sum() for val in model.parameters())
            loss = loss_pred + l2lamb * l2_norm
            #print('Epoch {}: train loss: {}'.format(ep, loss.item()))
            loss.backward()
            optimizer.step()

    if eval:
        print("Evaluations on training data")
        _, predres = torch.max(model(torch.tensor(X1_train.values).float()), 1)
        res = predres.detach().numpy().flatten().round()
        print('\n',
              classification_report(y1_train, res, target_names=['(0)', '(1)', '(2)'], digits=3))
        print("Evaluations on testing data")
        _, predres = torch.max(model(torch.tensor(X1_test.values).float()), 1)
        res = predres.detach().numpy().flatten().round()
        print('\n', classification_report(y1_test, res, target_names=['(0)', '(1)', '(2)'], digits=3))

    if save_clf:
        torch.save(model, f'./{data_name}.pth')

    return model


def retrain_models(d, hidden_size=50, epochs=20):
    X_train = pd.DataFrame(data=np.concatenate((d.X1_train.values, d.X2_train.values), axis=0),
                           columns=d.X1_train.columns)
    y_train = pd.DataFrame(data=np.concatenate((d.y1_train.values, d.y2_train.values), axis=0),
                           columns=d.y1_train.columns)
    rt_models = []
    seed = 100
    for i in tqdm(range(10)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            rt_torch_model = train_clf(X_train, y_train, d.X1_test, d.y1_test, hidden_size, epochs=epochs, eval=False)
        rt_models.append(InnModel(d, rt_torch_model, hidden_size))
    return rt_models


def retrain_models_leave_some_out(d, hidden_size=50, epochs=20, linear=False):
    # get leave-one-out dataset
    leave_size = int(0.01 * d.X1_train.shape[0])
    drop_idxs = np.random.choice(d.X1_train.index, leave_size, replace=False)
    X1_train_leave = d.X1_train.drop(drop_idxs)
    y1_train_leave = d.y1_train.drop(drop_idxs)
    lo_models = []
    seed = 2
    for i in tqdm(range(10)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            lo_torch_model = train_clf(X1_train_leave, y1_train_leave, d.X1_test, d.y1_test, hidden_size, epochs=epochs,
                                       eval=False)
        lo_models.append(InnModel(d, lo_torch_model, hidden_size))
    return lo_models


# WRAPPER CLASS FOR TORCH MODELS
class InnModel(MLModel):
    def __init__(self, data, m, hidden_layer_sizes=50):
        super().__init__(data)
        self._model = m
        self._h = hidden_layer_sizes
        self.coefs_, self.intercepts_ = self.get_params()
        self._model_type = "ann" if hidden_layer_sizes is not None else "linear"

    @property
    def feature_input_order(self):
        return self.data.features

    @property
    def backend(self):
        return "pytorch"

    @property
    def raw_model(self):
        return self._model

    @property
    def n_layers_(self):
        l = 2 if self._h is None else 4
        return l

    @property
    def n_features_in_(self):
        return len(self.data.features)

    @property
    def hidden_layer_sizes(self):
        return self._h

    @property
    def model_type(self):
        return self._model_type

    def convert_to_np(self, x):
        Xt = x
        if isinstance(x, pd.core.frame.DataFrame):
            Xt = x.values
        return np.array(Xt).astype(np.float32)

    def predict(self, x):
        try:
            x = self.convert_to_np(x)
        except:
            pass
        _, predres = torch.max(self._model(torch.tensor(x).float()), 1)
        return predres.detach().numpy().flatten().round().astype(np.int64)

    def predict_proba(self, x):
        try:
            self._model.to('cpu')
        except:
            pass
        self._model.eval()
        x = self.convert_to_np(x)
        yhats = self._model(torch.tensor(x).float()).detach().numpy().flatten().astype(np.float64).reshape(1, -1)
        return np.concatenate((1 - yhats, yhats), axis=0).transpose()

    def get_params(self):
        w1 = None
        w2 = None
        w3 = None
        b1 = None
        b2 = None
        b3 = None
        for i, item in enumerate(self._model.parameters()):
            if i == 0:
                w1 = item.data.detach().numpy()
            if i == 1:
                b1 = item.data.detach().numpy()
            if i == 2:
                w2 = item.data.detach().numpy()
            if i == 3:
                b2 = item.data.detach().numpy()
            if i == 4:
                w3 = item.data.detach().numpy()
            if i == 5:
                b3 = item.data.detach().numpy()
        if w2 is None:
            return [w1, b1]
        w1 = w1.transpose()
        w2 = w2.transpose()
        w3 = w3.transpose()
        return [w1, w2, w3], [b1, b2, b3]

    def partial_fit(self, Xdf, ydf, batch_size=256):
        trainds1 = TrainingDataset(Xdf, ydf)
        params = {'batch_size': batch_size, 'shuffle': True}
        traindl1 = DataLoader(trainds1, **params)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        epochs = 1
        l2lamb = 0.0001
        for ep in range(epochs):
            for (Xb, yb, _) in traindl1:
                self._model.train()
                optimizer.zero_grad()
                yp = self._model(torch.tensor(Xb).float())
                loss_pred = loss_fn(yp, torch.tensor(yb).long().view(-1))
                # print('Epoch {}: train loss: {}'.format(ep, loss.item()))
                l2_norm = sum(val.pow(2.0).sum() for val in self._model.parameters())
                loss = loss_pred + l2lamb * l2_norm
                loss.backward()
                optimizer.step()
        # update params
        self.coefs_, self.intercepts_ = self.get_params()


def get_flattened_weight_and_bias(model):
    w_all = model.get_params()[0]
    w_concat = np.append(w_all[0].flatten(), w_all[1].flatten())
    w_concat = np.append(w_concat, w_all[2].flatten())
    b_all = model.get_params()[1]
    b_concat = np.append(b_all[0].flatten(), b_all[1].flatten())
    b_concat = np.append(b_concat, b_all[2].flatten())
    wb_concat = np.append(w_concat, b_concat)
    return wb_concat


def inf_norm(x, y):
    return np.max(abs(x - y))


def cross_validation(d, num_h_neurons=20, epochs=50, data_name="adult"):
    torch.manual_seed(5)
    np.random.seed(1)
    random_state = check_random_state(10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    accs = []
    kf = KFold(5)
    for i, (train_index, test_index) in enumerate(kf.split(d.X1)):
        with HiddenPrints():
            torch_model = train_clf(d.X1.iloc[train_index], d.y1.iloc[train_index], d.X1_test, d.y1_test, num_h_neurons,
                                    epochs=epochs,
                                    data_name=data_name, save_clf=False,
                                    load_clf=False)
        model = InnModel(d, torch_model, num_h_neurons)
        accs.append(accuracy_score(d.y1.iloc[test_index], model.predict(d.X1.iloc[test_index])))
    print(f"accuracy: {np.array(accs).mean()}+-{np.array(accs).std()}")


def get_retrained_models_and_validation_set(d, model, num_h_neurons=20, epochs=20):
    X_train = pd.DataFrame(data=np.concatenate((d.X1_train.values, d.X2_train.values), axis=0), columns=d.X1.columns)
    y_train = pd.DataFrame(data=np.concatenate((d.y1_train.values, d.y2_train.values), axis=0), columns=d.y1.columns)
    # get retrained model - retrained
    rt_models = []
    seed = 100
    for i in tqdm(range(5)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            rt_torch_model = train_clf(X_train, y_train, d.X1_test, d.y1_test, num_h_neurons, epochs=epochs, eval=False)
        rt_models.append(InnModel(d, rt_torch_model, num_h_neurons))

    # get leave-one-out dataset
    leave_size = int(0.01 * d.X1_train.shape[0])
    drop_idxs = np.random.choice(d.X1_train.index, leave_size, replace=False)
    X1_train_leave = d.X1_train.drop(drop_idxs)
    y1_train_leave = d.y1_train.drop(drop_idxs)
    seed = 2
    for i in tqdm(range(5)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            lo_torch_model = train_clf(X1_train_leave, y1_train_leave, d.X1_test, d.y1_test, num_h_neurons,
                                       epochs=epochs)
        rt_models.append(InnModel(d, lo_torch_model, num_h_neurons))

    # validation set
    val_all = d.X1_train.values[model.predict(d.X1_train) == 0]
    val_y_all = d.X1_train.values[model.predict(d.X1_train) == 0]
    idxs = np.random.choice(val_all.shape[0], min(50, val_all.shape[0]), replace=False)
    val_set = val_all[idxs]
    val_y_set = val_y_all[idxs]
    return rt_models, val_set, val_y_set


def get_retrained_models_all(d, model, num_h_neurons=10, epochs=10):
    X_train = pd.DataFrame(data=np.concatenate((d.X1_train.values, d.X2_train.values), axis=0), columns=d.X1.columns)
    y_train = pd.DataFrame(data=np.concatenate((d.y1_train.values, d.y2_train.values), axis=0), columns=d.y1.columns)
    # get completely retrained model
    rt_models = []
    seed = 666
    for i in tqdm(range(5)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            rt_torch_model = train_clf(X_train, y_train, d.X1_test, d.y1_test, num_h_neurons, epochs=epochs, eval=False)
        rt_models.append(InnModel(d, rt_torch_model, num_h_neurons))

    # get leave-one-out dataset retrained model
    leave_size = int(0.01 * d.X1_train.shape[0])
    drop_idxs = np.random.choice(d.X1_train.index, leave_size, replace=False)
    X1_train_leave = d.X1_train.drop(drop_idxs)
    y1_train_leave = d.y1_train.drop(drop_idxs)
    seed = 2
    for i in tqdm(range(5)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            lo_torch_model = train_clf(X1_train_leave, y1_train_leave, d.X1_test, d.y1_test, num_h_neurons,
                                       epochs=epochs)
        rt_models.append(InnModel(d, lo_torch_model, 20))

    # get incrementally retrained models
    for i in range(5):
        it_model = copy.deepcopy(model)
        d2_size = int(d.y2_train.shape[0] * 0.1)
        # print(d2_size)
        idxs = np.random.choice(d.y2_train.shape[0], d2_size, replace=False)
        it_model.partial_fit(d.X2_train.values[idxs], d.y2_train.values[idxs], batch_size=256)
        rt_models.append(it_model)

    return rt_models
