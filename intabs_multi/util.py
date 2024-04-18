import numpy as np
import pandas as pd
from gurobipy import GurobiError
from sklearn.neighbors import LocalOutlierFactor

from intabs_multi.intabs_multi import UtilDataset, build_dataset_feature_types, build_inn_nodes, \
    build_inn_weights_biases, Inn, OptSolver


def build_inn_utils_multi(d, model, delta):
    utildataset = UtilDataset(len(d.columns) - 1, d.X1.values.shape[1],
                              build_dataset_feature_types(d.columns, d.ordinal_features, d.discrete_features,
                                                          d.continuous_features), d.feat_var_map)
    nodes = build_inn_nodes(model, model.n_layers_)
    weights, biases = build_inn_weights_biases(model, model.n_layers_, delta, nodes)
    inn_delta = Inn(model.n_layers_, delta, nodes, weights, biases)
    return utildataset, inn_delta


def delta_robustness_test_one_point(d, m, delta, ce, y_target):
    utildataset, inn_delta = build_inn_utils_multi(d, m, delta)
    delta_robust = False
    this_solver = OptSolver(utildataset, inn_delta, y_target, ce, mode=1, M=10000, x_prime=ce)
    try:
        if this_solver.robustness_test():
            delta_robust = True
    except GurobiError:
        pass
    return delta_robust


def delta_robustness_test(d, model, delta, found_ces, y_target):
    robust = 0
    for i, ce in enumerate(found_ces):
        if delta_robustness_test_one_point(d, model, delta, ce, y_target):
            robust += 1
    return robust / len(found_ces)


def eval_empirical_robustness(rt_models, ces, y_target):
    total_labels = len(ces) * len(rt_models) * y_target
    total_predicted = 0
    for m in rt_models:
        total_predicted += np.sum(m.predict(ces))
    return total_labels == total_predicted


def get_test_inputs(d, model, num_inputs=100, random_seed=0):
    np.random.seed(random_seed)
    test_all = d.pX.values[model.predict(d.pX) == 0]
    test_y_all = d.py.values[model.predict(d.pX) == 0]
    idxs = np.random.choice(test_all.shape[0], min(num_inputs, test_all.shape[0]), replace=False)
    test_set = test_all[idxs]
    test_y_set = test_y_all[idxs]
    test_set_df = pd.DataFrame(data=test_set, columns=d.X1.columns)
    test_set_full_df = pd.DataFrame(
        data=np.concatenate((test_set, test_y_set), axis=1),
        columns=d.columns)
    return test_set, test_set_df, test_set_full_df


def normalised_l1(xp, x):
    return np.sum(np.abs(xp - x)) / (xp.shape[0])


def eval_ces(inputs, ces, d, model, delta_val, retrained_models, y_target=2):
    # counterfactual point could be None or could contain NAN
    found, cost, validity_m2, validity_delta_inc, validity_delta_val, lof_score = 0, 0, 0, 0, 0, 0
    # ignore any NaN
    found_ces = []
    for item in ces:
        if item is None or np.isnan(item).any():
            continue
        found_ces.append(item)
    found = len(found_ces) / len(ces)

    # m2 validity
    for m in retrained_models:
        validity_m2 += np.sum(m.predict(found_ces) == y_target) / len(found_ces)
    validity_m2 /= len(retrained_models)

    # delta robustness
    validity_delta_val = delta_robustness_test(d, model, delta_val, found_ces, y_target)
    for i, ce in enumerate(found_ces):
        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit(np.concatenate((ce.reshape(1, -1), (d.X1[d.y1.values == 1]).values), axis=0))
        lof_score += -1 * lof.negative_outlier_factor_[0]
        cost += normalised_l1(ce, inputs[i])
    cost /= len(found_ces)
    lof_score /= len(found_ces)
    return [found, cost, lof_score, validity_m2, validity_delta_val]

