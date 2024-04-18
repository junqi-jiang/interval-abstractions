# baselines for computing CE, each function takes in the dataset, classifier, test inputs, with additional arguments
# and return CEs
from copy import copy

import numpy as np
from sklearn.utils import check_random_state
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import datetime
from intabs.classifier_binary import HiddenPrints
from carla.recourse_methods import Roar
import carla.recourse_methods.catalog as recourse_catalog
from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric

from intabs.evaluation import build_inn_utils, delta_robustness_test_one_point, delta_robustness_test, get_test_inputs, \
    eval_ces
from intabs.intabs import OptSolver, DataType

from alibi.explainers import cfproto
from rbr.rbr import RBR
from tabulate import tabulate


class CostLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CostLoss, self).__init__()

    def forward(self, x1, x2):
        dist = torch.abs(x1 - x2)
        return dist


def compute_wac(x, m, lamb=0.1, lr=0.01, max_iter=2000, max_allowed_minutes=0.5, target_class_prob=0.5):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialise the counterfactual search at the input point
    x = x.to(DEVICE)
    wac = Variable(x.clone(), requires_grad=True).to(DEVICE).float()

    # initialise an optimiser for gradient descent over the wac counterfactual point
    optimiser = Adam([wac], lr, amsgrad=True)

    # instantiate the two components of the loss function
    validity_loss = torch.nn.MSELoss()
    cost_loss = CostLoss()
    y_target = torch.Tensor([target_class_prob]).to(DEVICE)
    # compute class probability
    class_prob = m(wac.float())
    wac_valid = False
    iterations = 0
    if class_prob >= target_class_prob:
        wac_valid = True
    # set maximum allowed time for computing 1 counterfactual
    t0 = datetime.datetime.now()
    t_max = datetime.timedelta(minutes=max_allowed_minutes)
    # start gradient descent
    while not wac_valid or iterations <= max_iter:
        optimiser.zero_grad()
        class_prob = m(wac.float())
        wac_loss = validity_loss(class_prob, y_target) + lamb * cost_loss(x, wac)
        wac_loss.sum().backward()
        optimiser.step()
        # break conditions
        if class_prob >= target_class_prob:
            wac_valid = True
        if datetime.datetime.now() - t0 > t_max:
            break
        iterations += 1
    return wac


def gce(test_xs, model, lr=0.01, lamb=0.01, max_allowed_minutes=0.05, target_class_prob=0.5):
    ces = []
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = copy(model.raw_model).to(DEVICE).float()
    for i, x in enumerate(test_xs):
        x = torch.from_numpy(x).float()
        this_ce = compute_wac(x, m, lamb=lamb, lr=lr, max_iter=1000,
                              max_allowed_minutes=max_allowed_minutes,
                              target_class_prob=target_class_prob + 0.03).cpu().detach().numpy()
        ces.append(this_ce)
    m.to("cpu")
    return ces


def mce(test_set, d, model):
    utildataset, inn_delta_0 = build_inn_utils(d, model, 0)
    mce_ces = []
    for i, x in enumerate(test_set):
        mce_ces.append(OptSolver(utildataset, inn_delta_0, 1, x, mode=0, eps=0.05, M=1000).compute_counterfactual())
    return mce_ces


def face(test_set_full_df, model):
    hyperparams_face = {"mode": "knn", "fraction": 0.1}
    face = recourse_catalog.Face(model, hyperparams_face)
    face_ces = face.get_counterfactuals(test_set_full_df).values
    return face_ces


def nnce(test_set, d, model):
    X1_class1_clf = d.pX.values[model.predict(d.pX.values) == 1]
    X_class1_clf_robust = X1_class1_clf
    dist = DistanceMetric.get_metric("manhattan")
    tree = KDTree(X_class1_clf_robust, leaf_size=40, metric=dist)
    idxs = np.array(tree.query(test_set)[1]).flatten()
    tree_ces = X_class1_clf_robust[idxs]
    return tree_ces


def roar(test_set_full_df, model, lr=0.01, lambda_=0.01, delta_max=0.05):
    roar_params = {
        "lr": lr,
        "lambda_": lambda_,
        "delta_max": delta_max,
        "norm": 1,
        "t_max_min": 1,
        "loss_type": "MSE",
        "y_target": [1],
        "binary_cat_features": True,
        "loss_threshold": 1e-3,
        "discretize": False,
        "sample": False,
        "lime_seed": 10,
        "seed": 100,
    }
    if model._h is None:
        roar_params = {
            "lr": lr,
            "lambda_": lambda_,
            "delta_max": delta_max,
            "norm": 1,
            "t_max_min": 1,
            "loss_type": "MSE",
            "y_target": [1],
            "binary_cat_features": False,
            "loss_threshold": 1e-3,
            "discretize": False,
            "sample": False,
            "lime_seed": 0,
            "seed": 0,
        }
    roar = Roar(model, roar_params)
    return roar.get_counterfactuals(test_set_full_df).values


def counterfactual_stability(xp, model):
    # use std=0.1 (variance=0.01), k=1000 as in the original paper
    score_x = model.predict_proba(xp)[0][1]
    gaussian_samples = np.random.normal(xp, 0.1, (1000, len(xp)))
    model_scores = model.predict_proba(gaussian_samples)[:, 1]
    return np.sum((model_scores - np.abs(model_scores - score_x)) / len(model_scores))


def stablece_one(test_set, d, model, threshold=0.45):
    X_class1_clf = d.pX.values[model.predict(d.pX) == 1]
    if len(X_class1_clf) > 8000:
        np.random.shuffle(X_class1_clf)
        X_class1_clf = X_class1_clf[:8000]
    dist = DistanceMetric.get_metric("manhattan")
    cecestree = KDTree(X_class1_clf, leaf_size=40, metric=dist)
    ccf_ces = []
    for i, x in tqdm(enumerate(test_set)):
        # original prediction's class probability for class 1, scalar
        for k in range(1, len(X_class1_clf)):
            nn1 = X_class1_clf[cecestree.query(x.reshape(1, -1), k=k)[1].flatten()[-1]]
            if counterfactual_stability(nn1, model) >= threshold:
                ccf_ces.append(nn1)
                break
    return ccf_ces


def stablece(test_set, d, model, rt_models_validation):
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    stable_ces = None
    best_val = 0
    for threshold in thresholds:
        m2_validity = 0
        stablece_ces = stablece_one(test_set, d, model, threshold=threshold)
        if len(stablece_ces) == 0:
            break
        for m in rt_models_validation:
            m2_validity += np.sum(m.predict(stablece_ces) == 1) / len(stablece_ces)
        m2_validity /= len(rt_models_validation)
        if m2_validity > best_val:
            stable_ces = stablece_ces
            best_val = m2_validity
        if best_val >= 1:
            break
    return stable_ces


def rbr(test_set, d, model):
    random_state = check_random_state(0)
    delta_plus = 1.0
    epsilon_op = 1.0
    epsilon_pe = 1.0
    num_samples = 200
    rbr_ces = []
    for idx, x0 in tqdm(enumerate(test_set)):
        # x_ar, report = method.generate_recourse(x0, model, random_state, params)
        arg = RBR(model, d.X1_train.values, num_cfacts=1000, max_iter=500, random_state=random_state, device='cpu')
        x_ar = arg.fit_instance(x0, num_samples, 0.2, delta_plus, 1.0, epsilon_op, epsilon_pe, None)
        rbr_ces.append(x_ar)
    return rbr_ces


def mce_r(test_set, d, model, delta=0.05):
    utildataset, inn_delta_0 = build_inn_utils(d, model, 0)
    mce_ces = []
    for i, x in tqdm(enumerate(test_set)):
        eps = 0.05
        this_ce = OptSolver(utildataset, inn_delta_0, 1, x, mode=0, eps=eps, M=1000).compute_counterfactual()
        while eps < 20:
            this_ce = OptSolver(utildataset, inn_delta_0, 1, x, mode=0, eps=eps, M=1000).compute_counterfactual()
            if delta_robustness_test_one_point(d, model, delta, this_ce):
                break
            eps += 0.2
        mce_ces.append(this_ce)
    return mce_ces


def gce_r(test_set, d, model, delta=0.05):
    gce_ces = []
    target_probs = np.concatenate((np.arange(0.52, 1.00, 0.1), [0.97]))
    for i, x in tqdm(enumerate(test_set)):
        with HiddenPrints():
            target_prob = target_probs[0]
            this_input = test_set[i:i + 1]
            this_ce = gce(this_input, model, target_class_prob=target_prob)[0]
            for j, p in enumerate(target_probs[1:]):
                this_ce = gce(this_input, model, target_class_prob=p)[0]
                if delta_robustness_test_one_point(d, model, delta, this_ce):
                    break
            gce_ces.append(this_ce)
    return gce_ces


def run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=10., kap=0.1):
    cf = cfproto.CounterFactualProto(predict_fn, shape, use_kdtree=True, feature_range=(
        np.array(d.X1.values.min(axis=0)).reshape(1, -1), np.array(d.X1.values.max(axis=0)).reshape(1, -1)),
                                     cat_vars=cat_var,
                                     ohe=False, theta=theta, kappa=kap, c_steps=10)
    cf.fit(d.X1.values)
    this_point = x
    with HiddenPrints():
        explanation = cf.explain(this_point.reshape(1, -1), Y=None, target_class=None, k=20, k_type='mean',
                                 threshold=0., verbose=True, print_every=100, log_every=100)
    if explanation is None:
        return None
    if explanation["cf"] is None:
        return None
    proto_cf = explanation["cf"]["X"]
    proto_cf = proto_cf[0]
    return np.array(proto_cf)


def proto_r(test_set, d, model, delta=0.05, theta=10., plain=False, plain_proto_ces=None):
    utildataset, inn_delta_tar = build_inn_utils(d, model, delta)
    data_point = np.array(d.X1.values[1])
    shape = (1,) + data_point.shape[:]
    predict_fn = lambda x: model.predict_proba(x)
    cat_var = {}
    for idx in utildataset.feature_types:
        if utildataset.feature_types[idx] != DataType.CONTINUOUS_REAL:
            for varidx in utildataset.feat_var_map[idx]:
                cat_var[varidx] = 2
    CEs = []
    for i, x in tqdm(enumerate(test_set)):
        if delta >= 0.05:
            kaps = [0.85, 0.94, 0.99]
        else:
            kaps = [0.7, 0.85, 0.94]
        # run proto plain first
        if plain:
            best_cf = run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=0, kap=0.1)
            CEs.append(best_cf)
            continue
        else:
            if plain_proto_ces is None:
                best_cf = run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=0, kap=0.1)
            else:
                best_cf = plain_proto_ces[i]
        if best_cf is None:
            best_cf = x
            best_bound = -10000
        else:
            # test for Delta robustness
            target_solver = OptSolver(utildataset, inn_delta_tar, 1, x, mode=1, eps=0.01, x_prime=best_cf)
            found, bound = target_solver.compute_inn_bounds()
            if found == 1:
                CEs.append(best_cf)
                continue
            best_bound = bound if bound is not None else -10000
        for kappa in kaps:
            # print("using kappa ", kappa)
            with HiddenPrints():
                this_cf = run_proto_robust_one(d, x, predict_fn, shape, cat_var, theta=0, kap=kappa)
            if this_cf is None:
                # print(f"no cf using kappa {kappa}")
                continue
            target_solver = OptSolver(utildataset, inn_delta_tar, 1, x, mode=1, eps=0.01, x_prime=this_cf)
            found, bound = target_solver.compute_inn_bounds()
            if bound is None:
                continue
            if bound >= best_bound:
                best_cf = this_cf
                best_bound = bound
                if found == 1:
                    break
        CEs.append(best_cf)
    return CEs


def rnce(test_set, d, model, delta=0.05):
    utildataset, inn_delta = build_inn_utils(d, model, delta)
    X_class1_clf = d.pX.values[model.predict(d.pX) == 1]
    if len(X_class1_clf) > 12000:
        np.random.shuffle(X_class1_clf)
        X_class1_clf = X_class1_clf[:12000]
    valids = []
    for i, x in tqdm(enumerate(X_class1_clf)):
        lb_solver = OptSolver(utildataset, inn_delta, 1, x, mode=1, M=10000, x_prime=x)
        found, lb = lb_solver.compute_inn_bounds()
        if found == 1:
            valids.append(i)
    X_class1_clf_robust = X_class1_clf[valids]

    dist = DistanceMetric.get_metric("manhattan")
    treer = KDTree(X_class1_clf_robust, leaf_size=40, metric=dist)
    idxs = np.array(treer.query(test_set)[1]).flatten()
    return X_class1_clf_robust[idxs]


def optimise_distance(x, cfx, delta, d, model):
    opt = cfx
    for a in np.arange(0.05, 1.01, 0.05):
        cfxp = (1 - a) * x + (a) * cfx
        if delta_robustness_test_one_point(d, model, delta, cfxp):
            opt = cfxp
            break
    return opt


def rnce_opt(test_set, rnce_ces, d, model, delta_target):
    ces = []
    for i, ce in tqdm(enumerate(rnce_ces)):
        ces.append(optimise_distance(test_set[i], ce, delta_target, d, model))
    return ces


def run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=1000, run_id=1):
    test_set, test_set_df, test_set_full_df = get_test_inputs(d, model, 20, random_seed=random_seed)
    print(f"===== running experiments {run_id}/5 =====")
    print("===== running non-robust baselines 1-4/17 =====")

    # non-robust baselines
    gce_ces = gce(test_set, model)
    gce_scores = eval_ces(test_set, gce_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%
    mce_ces = mce(test_set, d, model)
    mce_scores = eval_ces(test_set, mce_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%
    proto_ces = proto_r(test_set, d, model, plain=True)
    proto_scores = eval_ces(test_set, proto_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%
    nnce_ces = nnce(test_set, d, model)
    nnce_scores = eval_ces(test_set, nnce_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%

    print("===== running robust baselines 5-7/17 =====")
    # run robust baselines
    roar_ces = roar(test_set_full_df, model, lr=0.02, lambda_=0.001, delta_max=inc_delta)
    roar_scores = eval_ces(test_set, roar_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%
    rbr_ces = rbr(test_set, d, model)
    rbr_scores = eval_ces(test_set, rbr_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%
    stable_ces = stablece(test_set, d, model, rt_models_validation=rt_models_validation)
    stable_scores = eval_ces(test_set, stable_ces, d, model, inc_delta, val_delta, rt_models_eval)

    print("===== running our methods 8-12/17, incremental delta =====")
    # our methods: target delta: incremental retraining delta
    gcer_ces = gce_r(test_set, d, model, delta=inc_delta)
    gcer_scores = eval_ces(test_set, gcer_ces, d, model, inc_delta, val_delta, rt_models_eval)
    mcer_ces = mce_r(test_set, d, model, delta=inc_delta)
    mcer_scores = eval_ces(test_set, mcer_ces, d, model, inc_delta, val_delta, rt_models_eval)
    protor_ces = proto_r(test_set, d, model, delta=inc_delta, plain=False, plain_proto_ces=proto_ces)
    protor_scores = eval_ces(test_set, protor_ces, d, model, inc_delta, val_delta, rt_models_eval)

    rnce_ces = rnce(test_set, d, model, delta=inc_delta)
    rnce_scores = eval_ces(test_set, rnce_ces, d, model, inc_delta, val_delta, rt_models_eval)

    rnceopt_ces = rnce_opt(test_set, rnce_ces, d, model, delta_target=inc_delta)
    rnceopt_scores = eval_ces(test_set, rnceopt_ces, d, model, inc_delta, val_delta, rt_models_eval)

    print("===== running our methods 13-17/17, validation delta =====")
    # our methods: target delta: validation delta
    gcer_ces_val = gce_r(test_set, d, model, delta=val_delta)
    gcer_scores_val = eval_ces(test_set, gcer_ces_val, d, model, inc_delta, val_delta, rt_models_eval)
    mcer_ces_val = mce_r(test_set, d, model, delta=val_delta)
    mcer_scores_val = eval_ces(test_set, mcer_ces_val, d, model, inc_delta, val_delta, rt_models_eval)

    protor_ces_val = proto_r(test_set, d, model, delta=val_delta, plain=False, plain_proto_ces=proto_ces)
    protor_scores_val = eval_ces(test_set, protor_ces_val, d, model, inc_delta, val_delta, rt_models_eval)

    rnce_ces_val = rnce(test_set, d, model, delta=val_delta)
    rnce_scores_val = eval_ces(test_set, rnce_ces_val, d, model, inc_delta, val_delta, rt_models_eval)

    rnceopt_ces_val = rnce_opt(test_set, rnce_ces_val, d, model, delta_target=val_delta)
    rnceopt_scores_val = eval_ces(test_set, rnceopt_ces_val, d, model, inc_delta, val_delta, rt_models_eval)

    return [gce_scores, mce_scores, proto_scores, nnce_scores, roar_scores, rbr_scores, stable_scores, gcer_scores,
            mcer_scores, protor_scores, rnce_scores, rnceopt_scores, gcer_scores_val, mcer_scores_val,
            protor_scores_val, rnce_scores_val, rnceopt_scores_val]


def run_exps(res_1, res_2, res_3, res_4, res_5):
    # res_1 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=1050,
    #                           run_id=1)
    # res_2 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=7050,
    #                           run_id=2)
    # res_3 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=3050,
    #                           run_id=3)
    # res_4 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=3990,
    #                           run_id=4)
    # res_5 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=6666,
    #                           run_id=5)
    mean_res = np.mean((res_1, res_2, res_3, res_4, res_5), axis=0)
    std_res = np.std((res_1, res_2, res_3, res_4, res_5), axis=0)
    print("average results")
    scores_names = ["name", "coverage", "cost", "lof", "vm2", "vdelta-val", "vdelta-inc"]
    scores_table = [scores_names,
                    np.concatenate((['gce'], np.round(mean_res[0], 3))),
                    np.concatenate((['mce'], np.round(mean_res[1], 3))),
                    np.concatenate((['proto'], np.round(mean_res[2], 3))),
                    np.concatenate((['nnce'], np.round(mean_res[3], 3))),
                    np.concatenate((['roar'], np.round(mean_res[4], 3))),
                    np.concatenate((['rbr'], np.round(mean_res[5], 3))),
                    np.concatenate((['stable-ce'], np.round(mean_res[6], 3))),
                    np.concatenate((['gce-r'], np.round(mean_res[7], 3))),
                    np.concatenate((['mce-r'], np.round(mean_res[8], 3))),
                    np.concatenate((['proto-r'], np.round(mean_res[9], 3))),
                    np.concatenate((['rnce'], np.round(mean_res[10], 3))),
                    np.concatenate((['rnce-opt'], np.round(mean_res[11], 3))),
                    np.concatenate((['gce-r-val'], np.round(mean_res[12], 3))),
                    np.concatenate((['mce-r-val'], np.round(mean_res[13], 3))),
                    np.concatenate((['proto-r-val'], np.round(mean_res[14], 3))),
                    np.concatenate((['rnce-val'], np.round(mean_res[15], 3))),
                    np.concatenate((['rnce-opt-val'], np.round(mean_res[16], 3)))]
    print(tabulate(scores_table, headers='firstrow', tablefmt='outline'))

    print("std results")
    scores_table = [scores_names,
                    np.concatenate((['gce'], np.round(std_res[0], 5))),
                    np.concatenate((['mce'], np.round(std_res[1], 5))),
                    np.concatenate((['proto'], np.round(std_res[2], 5))),
                    np.concatenate((['nnce'], np.round(std_res[3], 5))),
                    np.concatenate((['roar'], np.round(std_res[4], 5))),
                    np.concatenate((['rbr'], np.round(std_res[5], 5))),
                    np.concatenate((['stable-ce'], np.round(std_res[6], 5))),
                    np.concatenate((['gce-r'], np.round(std_res[7], 5))),
                    np.concatenate((['mce-r'], np.round(std_res[8], 5))),
                    np.concatenate((['proto-r'], np.round(std_res[9], 5))),
                    np.concatenate((['rnce'], np.round(std_res[10], 5))),
                    np.concatenate((['rnce-opt'], np.round(std_res[11], 5))),
                    np.concatenate((['gce-r-val'], np.round(std_res[12], 5))),
                    np.concatenate((['mce-r-val'], np.round(std_res[13], 5))),
                    np.concatenate((['proto-r-val'], np.round(std_res[14], 5))),
                    np.concatenate((['rnce-val'], np.round(std_res[15], 5))),
                    np.concatenate((['rnce-opt-val'], np.round(std_res[16], 5)))]
    print(tabulate(scores_table, headers='firstrow', tablefmt='outline'))
    return mean_res, std_res


def run_exps_all_once_lr(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=1000, run_id=1):
    test_set, test_set_df, test_set_full_df = get_test_inputs(d, model, 20, random_seed=random_seed)
    print(f"===== running experiments {run_id}/5 =====")
    # non-robust baselines
    nnce_ces = nnce(test_set, d, model)
    nnce_scores = eval_ces(test_set, nnce_ces, d, model, inc_delta, val_delta, rt_models_eval)
    # %%
    # run robust baselines
    max_delta = inc_delta if inc_delta >= val_delta else val_delta
    roar_ces = roar(test_set_full_df, model, lr=0.02, lambda_=0.001, delta_max=inc_delta)
    roar_scores = eval_ces(test_set, roar_ces, d, model, inc_delta, val_delta, rt_models_eval)
    print("===== running our methods 8-12/17, incremental delta =====")

    rnce_ces = rnce(test_set, d, model, delta=inc_delta)
    rnce_scores = eval_ces(test_set, rnce_ces, d, model, inc_delta, val_delta, rt_models_eval)

    rnceopt_ces = rnce_opt(test_set, rnce_ces, d, model, delta_target=inc_delta)
    rnceopt_scores = eval_ces(test_set, rnceopt_ces, d, model, inc_delta, val_delta, rt_models_eval)

    print("===== running our methods 13-17/17, validation delta =====")
    rnce_ces_val = rnce(test_set, d, model, delta=val_delta)
    rnce_scores_val = eval_ces(test_set, rnce_ces_val, d, model, inc_delta, val_delta, rt_models_eval)

    rnceopt_ces_val = rnce_opt(test_set, rnce_ces_val, d, model, delta_target=val_delta)
    rnceopt_scores_val = eval_ces(test_set, rnceopt_ces_val, d, model, inc_delta, val_delta, rt_models_eval)

    return [nnce_scores, roar_scores, rnce_scores, rnceopt_scores, rnce_scores_val, rnceopt_scores_val]



def run_exps_lr(res_1, res_2, res_3, res_4, res_5):
    # res_1 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=1050,
    #                           run_id=1)
    # res_2 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=7050,
    #                           run_id=2)
    # res_3 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=3050,
    #                           run_id=3)
    # res_4 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=3990,
    #                           run_id=4)
    # res_5 = run_exps_all_once(d, model, rt_models_eval, rt_models_validation, inc_delta, val_delta, random_seed=6666,
    #                           run_id=5)
    mean_res = np.mean((res_1, res_2, res_3, res_4, res_5), axis=0)
    std_res = np.std((res_1, res_2, res_3, res_4, res_5), axis=0)
    print("average results")
    scores_names = ["name", "coverage", "cost", "lof", "vm2", "vdelta-val", "vdelta-inc"]
    scores_table = [scores_names,
                    np.concatenate((['nnce'], np.round(mean_res[0], 3))),
                    np.concatenate((['roar'], np.round(mean_res[1], 3))),
                    np.concatenate((['rnce'], np.round(mean_res[2], 3))),
                    np.concatenate((['rnce-opt'], np.round(mean_res[3], 3))),
                    np.concatenate((['rnce-val'], np.round(mean_res[4], 3))),
                    np.concatenate((['rnce-opt-val'], np.round(mean_res[5], 3)))]
    print(tabulate(scores_table, headers='firstrow', tablefmt='outline'))

    print("std results")
    scores_table = [scores_names,
                    np.concatenate((['nnce'], np.round(std_res[0], 5))),
                    np.concatenate((['roar'], np.round(std_res[1], 5))),
                    np.concatenate((['rnce'], np.round(std_res[2], 5))),
                    np.concatenate((['rnce-opt'], np.round(std_res[3], 5))),
                    np.concatenate((['rnce-val'], np.round(std_res[4], 5))),
                    np.concatenate((['rnce-opt-val'], np.round(std_res[5], 5)))]
    print(tabulate(scores_table, headers='firstrow', tablefmt='outline'))
    return mean_res, std_res
