# baselines for computing CE, each function takes in the dataset, classifier, test inputs, with additional arguments
# and return CEs
from copy import copy

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric

from intabs_multi.util import build_inn_utils_multi, delta_robustness_test_one_point, get_test_inputs, eval_ces
from intabs_multi.intabs_multi import OptSolver, DataType

from tabulate import tabulate


def nnce(test_set, d, model, y_target):
    X1_class1_clf = d.pX.values[model.predict(d.pX.values) == y_target]
    X_class1_clf_robust = X1_class1_clf
    dist = DistanceMetric.get_metric("manhattan")
    tree = KDTree(X_class1_clf_robust, leaf_size=40, metric=dist)
    idxs = np.array(tree.query(test_set)[1]).flatten()
    tree_ces = X_class1_clf_robust[idxs]
    return tree_ces


def rnce(test_set, d, model, delta=0.05, y_target=2):
    utildataset, inn_delta = build_inn_utils_multi(d, model, delta)
    X_class1_clf = d.pX.values[model.predict(d.pX) == y_target]
    if len(X_class1_clf) > 12000:
        np.random.shuffle(X_class1_clf)
        X_class1_clf = X_class1_clf[:12000]
    valids = []
    for i, x in tqdm(enumerate(X_class1_clf)):
        lb_solver = OptSolver(utildataset, inn_delta, y_target, x, mode=1, M=10000, x_prime=x)
        if lb_solver.robustness_test():
            valids.append(i)
    X_class1_clf_robust = X_class1_clf[valids]

    dist = DistanceMetric.get_metric("manhattan")
    treer = KDTree(X_class1_clf_robust, leaf_size=40, metric=dist)
    idxs = np.array(treer.query(test_set)[1]).flatten()
    return X_class1_clf_robust[idxs]


def optimise_distance(x, cfx, delta, d, model, y_target=2):
    opt = cfx
    for a in np.arange(0.05, 1.01, 0.05):
        cfxp = (1 - a) * x + (a) * cfx
        if delta_robustness_test_one_point(d, model, delta, cfxp, y_target):
            opt = cfxp
            break
    return opt


def rnce_opt(test_set, rnce_ces, d, model, delta_target, y_target):
    ces = []
    for i, ce in tqdm(enumerate(rnce_ces)):
        ces.append(optimise_distance(test_set[i], ce, delta_target, d, model, y_target))
    return ces


def run_exps_all_once(d, model, rt_models_eval, val_delta, random_seed=1000, y_target=2):
    test_set, test_set_df, test_set_full_df = get_test_inputs(d, model, 20, random_seed=random_seed)

    nnce_ces = nnce(test_set, d, model, y_target=y_target)
    nnce_scores = eval_ces(test_set, nnce_ces, d, model, val_delta, rt_models_eval, y_target=y_target)
    # %%
    rnce_ces = rnce(test_set, d, model, delta=val_delta, y_target=y_target)
    rnce_scores = eval_ces(test_set, rnce_ces, d, model, val_delta, rt_models_eval, y_target=y_target)

    rnceopt_ces = rnce_opt(test_set, rnce_ces, d, model, delta_target=val_delta, y_target=y_target)
    rnceopt_scores = eval_ces(test_set, rnceopt_ces, d, model, val_delta, rt_models_eval, y_target=y_target)

    return [nnce_scores, rnce_scores, rnceopt_scores]


def run_all(d, model, rt_models_eval, val_delta):
    res_1 = run_exps_all_once(d, model, rt_models_eval, val_delta, random_seed=1000, y_target=2)
    res_2 = run_exps_all_once(d, model, rt_models_eval, val_delta, random_seed=5000, y_target=2)
    res_3 = run_exps_all_once(d, model, rt_models_eval, val_delta, random_seed=21030, y_target=2)
    res_4 = run_exps_all_once(d, model, rt_models_eval, val_delta, random_seed=1140, y_target=2)
    res_5 = run_exps_all_once(d, model, rt_models_eval, val_delta, random_seed=9456, y_target=2)
    scores_names = ["name", "coverage", "cost", "lof", "vm2", "vdelta-val"]
    mean_res = np.mean((res_1, res_2, res_3, res_4, res_5), axis=0)
    std_res = np.std((res_1, res_2, res_3, res_4, res_5), axis=0)
    print("average results")
    scores_table = [scores_names,
                    np.concatenate((['nnce'], np.round(mean_res[0], 3))),
                    np.concatenate((['rnce'], np.round(mean_res[1], 3))),
                    np.concatenate((['rnce-opt'], np.round(mean_res[2], 3)))]
    print(tabulate(scores_table, headers='firstrow', tablefmt='outline'))
    print("std results")
    scores_table = [scores_names,
                    np.concatenate((['nnce'], np.round(std_res[0], 3))),
                    np.concatenate((['rnce'], np.round(std_res[1], 3))),
                    np.concatenate((['rnce-opt'], np.round(std_res[2], 3)))]
    print(tabulate(scores_table, headers='firstrow', tablefmt='outline'))
