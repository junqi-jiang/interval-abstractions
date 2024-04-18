import copy
from matplotlib import pyplot as plt
from sklearn.metrics import DistanceMetric
from intabs.intabs import *
from intabs.classifier_binary import *
from sklearn.neighbors import KDTree, LocalOutlierFactor


def normalised_l1(xp, x):
    return np.sum(np.abs(xp - x)) / (xp.shape[0])


# determine Delta using incremental retraining
def find_percentage_of_val_delta(target_delta, percentages, delta_magnitudes):
    percentage_idx = 0
    opt_dist = 100
    for i, p in enumerate(percentages):
        if abs(delta_magnitudes[i] - target_delta) <= opt_dist:
            opt_dist = abs(delta_magnitudes[i] - target_delta)
            percentage_idx = i
    return percentages[percentage_idx]


def get_delta_incremental_training(d, model, batch_size=256):
    # get retrained model - incremental training
    magnitudes = [0]
    for size in tqdm(np.arange(0, 101, 2)):
        if size == 0:
            continue
        magnitudes.append(incremental_training_get_delta(d, model, retrain_size=size * 0.01, new_model_sizes=10,
                                                         batch_size=batch_size))
    return np.arange(0, 101, 2), magnitudes


def incremental_training_get_delta(d, model, retrain_size=0.05, new_model_sizes=10, batch_size=256):
    it_models = []
    wb_concat = get_flattened_weight_and_bias(model)
    avg_delta = 0
    for i in range(new_model_sizes):
        it_model = copy.deepcopy(model)
        d2_size = int(d.y2_train.shape[0] * retrain_size)
        # print(d2_size)
        idxs = np.random.choice(d.y2_train.shape[0], d2_size, replace=False)
        it_model.partial_fit(d.X2_train.values[idxs], d.y2_train.values[idxs], batch_size=batch_size)
        it_models.append(it_model)
        wb_concat_new = get_flattened_weight_and_bias(it_model)
        avg_delta += inf_norm(wb_concat, wb_concat_new)
    return avg_delta / new_model_sizes


def build_inn_utils(d, model, delta):
    utildataset = UtilDataset(len(d.columns) - 1, d.X1.values.shape[1],
                              build_dataset_feature_types(d.columns, d.ordinal_features, d.discrete_features,
                                                          d.continuous_features), d.feat_var_map)
    nodes = build_inn_nodes(model, model.n_layers_)
    weights, biases = build_inn_weights_biases(model, model.n_layers_, delta, nodes)
    inn_delta = Inn(model.n_layers_, delta, nodes, weights, biases)
    return utildataset, inn_delta


def rnce_for_delta(d, model, delta, val_set):
    utildataset, inn_delta = build_inn_utils(d, model, delta)
    X1_class1_clf = d.X1.values[model.predict(d.X1) == 1]
    if len(X1_class1_clf) > 10000:
        np.random.shuffle(X1_class1_clf)
        X1_class1_clf = X1_class1_clf[:10000]
    valids = []
    for i, x in tqdm(enumerate(X1_class1_clf)):
        lb_solver = OptSolver(utildataset, inn_delta, 1, x, mode=1, M=10000, x_prime=x)
        found, lb = lb_solver.compute_inn_bounds()
        if found == 1:
            valids.append(i)
    X_class1_clf_robust = X1_class1_clf[valids]

    dist = DistanceMetric.get_metric("manhattan")
    treer = KDTree(X_class1_clf_robust, leaf_size=40, metric=dist)
    idxs = np.array(treer.query(val_set)[1]).flatten()
    return X_class1_clf_robust[idxs]


def get_retrained_models_and_validation_set(d, model, num_h_neurons=20, epochs=20, linear=False):
    X_train = pd.DataFrame(data=np.concatenate((d.X1_train.values, d.X2_train.values), axis=0), columns=d.X1.columns)
    y_train = pd.DataFrame(data=np.concatenate((d.y1_train.values, d.y2_train.values), axis=0), columns=d.y1.columns)
    # get retrained model - retrained
    rt_models = []
    seed = 100
    for i in tqdm(range(5)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            rt_torch_model = train_clf(X_train, y_train, d.X1_test, d.y1_test, num_h_neurons, epochs=epochs, eval=False, linear=linear)
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
                                       epochs=epochs, linear=linear)
        rt_models.append(InnModel(d, lo_torch_model, num_h_neurons))

    # validation set
    val_all = d.X1_test.values[model.predict(d.X1_test) == 0]
    val_y_all = d.y1_test.values[model.predict(d.X1_test) == 0]
    idxs = np.random.choice(val_all.shape[0], min(50, val_all.shape[0]), replace=False)
    val_set = val_all[idxs]
    val_y_set = val_y_all[idxs]
    return rt_models, val_set, val_y_set


def test_delta_with_val_set(d, model, delta, val_set, rt_models):
    rnces = rnce_for_delta(d, model, delta, val_set)
    total_labels = len(val_set) * len(rt_models)
    total_predicted = 0
    for m in rt_models:
        total_predicted += np.sum(m.predict(rnces))
    return total_predicted >= total_labels


def plot_deltas(target_delta, inc_delta, percentages, delta_magnitudes, data_name):
    # plot
    fig, ax = plt.subplots()
    ax.plot(percentages, delta_magnitudes)
    ax.grid()
    target_delta_percentage = find_percentage_of_val_delta(target_delta, percentages, delta_magnitudes)
    plot_x = np.arange(0, 101, 25)
    plot_x = np.concatenate((plot_x, [10]))
    plot_x = np.concatenate((plot_x, [target_delta_percentage]))
    plot_x.sort()
    ax.set_xticks(plot_x)
    plot_x = plot_x.astype("object")
    plot_x[np.where(plot_x == 10)[0][0]] = "10"
    plot_x[np.where(plot_x == target_delta_percentage)[0][0]] = f"{target_delta_percentage}"
    ax.set_xticklabels(plot_x)
    ax.set_xlabel("retraining on a% of D2", fontsize=12)
    plt.xticks(fontsize=9)
    plot_y = list(ax.get_yticks())
    plot_y = np.concatenate((plot_y, [target_delta, inc_delta])).round(5)
    plot_y.sort()
    ax.set_yticks(plot_y)
    plot_y = plot_y.astype("object")
    plot_y[np.where(plot_y == target_delta)[0][0]] = f"{target_delta} " + r"($\delta_{val}$)"
    plot_y[np.where(plot_y == inc_delta.round(5))[0][0]] = f"{inc_delta.round(3)} " + r"($\delta_{inc}$)"
    ax.set_yticklabels(plot_y)
    ax.set_ylabel("$\delta$ values", fontsize=12)
    ax.scatter(target_delta_percentage, target_delta, marker='o', color='red')
    ax.scatter(10, inc_delta, marker='o', color='red')
    # fig.set_figheight(4)
    fig.tight_layout()
    # fig.set_figwidth(5)
    fig.savefig(f"./plots/delta_plots_{data_name}.png", dpi=300)


def get_test_inputs(d, model, num_inputs=100, random_seed=0):
    np.random.seed(random_seed)
    test_all = d.X2_test.values[model.predict(d.X2_test) == 0]
    test_y_all = d.y2_test.values[model.predict(d.X2_test) == 0]
    idxs = np.random.choice(test_all.shape[0], min(num_inputs, test_all.shape[0]), replace=False)
    test_set = test_all[idxs]
    test_y_set = test_y_all[idxs]
    test_set_df = pd.DataFrame(data=test_set, columns=d.X1.columns)
    test_set_full_df = pd.DataFrame(
        data=np.concatenate((test_set, test_y_set), axis=1),
        columns=d.columns)
    return test_set, test_set_df, test_set_full_df


def delta_robustness_test(d, m, delta, ces):
    utildataset, inn_delta = build_inn_utils(d, m, delta)
    num_delta_robust = 0
    for i, ce in enumerate(ces):
        this_solver = OptSolver(utildataset, inn_delta, 1, ce, mode=1, M=10000, x_prime=ce)
        try:
            found, _ = this_solver.compute_inn_bounds()
            if found == 1:
                num_delta_robust += 1
        except:
            pass
    return num_delta_robust / len(ces)


def delta_robustness_test_one_point(d, m, delta, ce):
    utildataset, inn_delta = build_inn_utils(d, m, delta)
    delta_robust = False
    this_solver = OptSolver(utildataset, inn_delta, 1, ce, mode=1, M=10000, x_prime=ce)
    try:
        found, bound = this_solver.compute_inn_bounds()
        if found == 1:
            delta_robust = True
    except GurobiError:
        pass
    return delta_robust


def delta_robustness_test_for_plots(d, model, ces, deltas_plot):
    delta_validities = []
    for delta in deltas_plot:
        delta_validities.append(delta_robustness_test(d, model, delta, ces))
    return delta_validities


def get_deltas_plot(inc_delta, val_delta):
    temp_arr = np.sort(np.concatenate((np.arange(0, inc_delta * 1.00001, inc_delta / 10), [val_delta])))
    res_arr = []
    for item in temp_arr:
        if item == val_delta:
            res_arr.append(item)
        else:
            if abs(item - val_delta) < 0.035 * inc_delta:
                continue
            res_arr.append(item)
    return np.array(res_arr)


def plot_delta_validity(deltas_plot, val_delta, gce_validity, mce_validity, proto_validity, roar_validity,
                        gcer_validity, mcer_validity, protor_validity,
                        rnce_validity,
                        robustified=True,
                        data_name="adult"):
    fig, ax = plt.subplots()
    if not robustified:
        ax.plot(deltas_plot, gce_validity, "b>--", label="GCE", alpha=0.9)
        ax.plot(deltas_plot, mce_validity, "m<--", label="MCE", alpha=0.9)
        ax.plot(deltas_plot, proto_validity, "cv--", label="PROTO", alpha=0.9)
        ax.plot(deltas_plot, roar_validity, "ro--", label="ROAR", alpha=0.9)
    else:
        ax.plot(deltas_plot, gce_validity, "b>--", label="GCE", alpha=0.9)
        ax.plot(deltas_plot, mce_validity, "m<--", label="MCE", alpha=0.9)
        ax.plot(deltas_plot, proto_validity, "cv--", label="PROTO", alpha=0.9)
        ax.plot(deltas_plot, gcer_validity, "b>-", label="GCE-R", alpha=0.9)
        ax.plot(deltas_plot, mcer_validity, "m<-", label="MCE-R", alpha=0.9)
        ax.plot(deltas_plot, protor_validity, "cv-", label="PROTO-R", alpha=0.9)
        ax.plot(deltas_plot, roar_validity, "ro--", label="ROAR", alpha=0.9)
        ax.plot(deltas_plot, rnce_validity, "P--", color="gold", label="RNCE", alpha=0.9)
    # transparent plot for better formatting
    ax.plot(deltas_plot, mce_validity, "ro--", label="ROAR", alpha=0)
    ax.set_xticks(deltas_plot)
    idx = 0
    labels = []
    for i in range(len(deltas_plot)):
        real_idx = round(deltas_plot[i] * 10 / deltas_plot[-1]) * 0.1
        if deltas_plot[i] == val_delta:
            labels.append("$\delta_{val}=$" + str(np.round(deltas_plot[i], 3)))
            continue
        if i == len(deltas_plot) - 1:
            labels.append("$\delta_{inc}=$" + str(np.round(deltas_plot[-1], 3)))
        elif i == 0:
            labels.append(str(0))
            idx += 0.1
        else:
            labels.append(f"{np.round(real_idx, 2)}" + "*$\delta_{inc}$")
            idx += 0.1
    ax.set_xticklabels(labels, fontsize=9)
    ylabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(ylabels)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.grid()
    # ax.legend()
    plt.xticks(rotation=60)
    ax.set_ylabel("$\Delta$-validity", fontsize=12)
    ax.set_xlabel("$\delta$ values", fontsize=12)
    fig.tight_layout()
    if robustified:
        fig.savefig(f"./plots/delta_validity_{data_name}_rob.png", dpi=300)
    else:
        fig.savefig(f"./plots/delta_validity_{data_name}.png", dpi=300)


def plot_delta_validity_lgd(deltas_plot, val_delta, gce_validity, mce_validity, proto_validity, roar_validity,
                            gcer_validity, protor_validity,
                            rnce_validity,
                            mcer_validity):
    fig, ax = plt.subplots()
    ax.plot(deltas_plot, gce_validity, "b>--", label="GCE", alpha=0.9)
    ax.plot(deltas_plot, gcer_validity, "b>-", label="GCE-R", alpha=0.9)
    ax.plot(deltas_plot, proto_validity, "cv--", label="PROTO", alpha=0.9)
    ax.plot(deltas_plot, protor_validity, "cv-", label="PROTO-R", alpha=0.9)
    ax.plot(deltas_plot, mce_validity, "m<--", label="MCE", alpha=0.9)
    ax.plot(deltas_plot, mcer_validity, "m<-", label="MCE-R", alpha=0.9)
    ax.plot(deltas_plot, roar_validity, "ro--", label="ROAR", alpha=0.9)
    ax.plot(deltas_plot, rnce_validity, "P--", color="gold", label="RNCE", alpha=0.9)
    ax.set_xticks(deltas_plot)
    ax.grid()
    lgd = ax.legend(bbox_to_anchor=(0., 1., 1.4, .2), loc='lower left',
                    ncol=8, mode="expand", borderaxespad=2)
    fig.tight_layout()
    ax.set_xticks(deltas_plot)
    idx = 0
    labels = []
    for i in range(len(deltas_plot)):
        if deltas_plot[i] == val_delta:
            labels.append("$\delta_{val}$")
            continue
        if i == len(deltas_plot) - 1:
            labels.append("$\delta_{inc}=$" + str(np.round(deltas_plot[-1], 3)))
        elif i == 0:
            labels.append(str(0))
            idx += 0.1
        else:
            labels.append(f"{np.round(idx, 2)}" + "*$\delta_{inc}$")
            idx += 0.1
    ax.set_xticklabels(labels, fontsize=9)
    ylabels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(ylabels)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.grid()
    # ax.legend()
    plt.xticks(rotation=60)
    ax.set_ylabel("$\Delta$-validity", fontsize=12)
    ax.set_xlabel("$\delta$ values", fontsize=12)
    fig.tight_layout()

    text = ax.text(-0.2, 1.05, "Aribitrary text", transform=ax.transAxes)
    fig.savefig("./plots/delta_validity_legend.png", dpi=500, bbox_extra_artists=(lgd, text), bbox_inches='tight')


def eval_ces(inputs, ces, d, model, delta_inc, delta_val, retrained_models):
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
        validity_m2 += np.sum(m.predict(found_ces) == 1) / len(found_ces)
    validity_m2 /= len(retrained_models)

    # delta robustness
    validity_delta_inc = delta_robustness_test(d, model, delta_inc, found_ces)
    validity_delta_val = delta_robustness_test(d, model, delta_val, found_ces)
    for i, ce in enumerate(found_ces):
        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit(np.concatenate((ce.reshape(1, -1), (d.X1[d.y1.values == 1]).values), axis=0))
        lof_score += -1 * lof.negative_outlier_factor_[0]
        cost += normalised_l1(ce, inputs[i])
    cost /= len(found_ces)
    lof_score /= len(found_ces)
    return [found, cost, lof_score, validity_m2, validity_delta_val, validity_delta_inc]


def get_retrained_models_all(d, model, num_h_neurons=10, epochs=10, linear=False):
    X_train = pd.DataFrame(data=np.concatenate((d.X1_train.values, d.X2_train.values), axis=0), columns=d.X1.columns)
    y_train = pd.DataFrame(data=np.concatenate((d.y1_train.values, d.y2_train.values), axis=0), columns=d.y1.columns)
    # get completely retrained model
    rt_models = []
    seed = 666
    for i in tqdm(range(5)):
        with HiddenPrints():
            seed += 1
            torch.manual_seed(seed)
            rt_torch_model = train_clf(X_train, y_train, d.X1_test, d.y1_test, num_h_neurons, epochs=epochs, eval=False, linear=linear)
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
                                       epochs=epochs, linear=linear)
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

