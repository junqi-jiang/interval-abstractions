import enum
from gurobipy import *
import numpy as np


class DataType(enum.Enum):
    DISCRETE = 0
    ORDINAL = 1
    CONTINUOUS_REAL = 2


class UtilDataset:
    """
        Dataset class providing the following information of the dataset
        (ordinal/discrete features are encoded, continuous features are min-max scaled so feature ranges are [0, 1])

        num_features: int, number of features in the original dataset before preprocessing
        num_variables: int, number of variables in the dataset after preprocessing
        feature_types: dict of {(int) feature index, (DataType) type of data}
        feat_var_map: dict of {(int) feature index, [corresponding variable indices]}

    """

    def __init__(self, num_features, num_var, feature_types, feat_var_map):
        self.num_features = num_features
        self.num_variables = num_var
        self.feature_types = feature_types
        self.feat_var_map = feat_var_map


class Interval:
    def __init__(self, value, lb=0, ub=0):
        self.value = value
        self.lb = lb
        self.ub = ub

    def set_bounds(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def get_bound(self, y_prime):
        return self.lb if y_prime is 1 else self.ub


class Node(Interval):
    def __init__(self, layer, index, lb=0, ub=0):
        super().__init__(lb, ub)
        self.layer = layer
        self.index = index
        self.loc = (layer, index)

    def __str__(self):
        return str(self.loc)



class Inn:
    """
    args:
    delta: weight shift value
    nodes: dict of {int layer num, [Node1, Node2, ...]}
    weights: dict of {(Node in prev layer, Node in this layer), Interval}.
    biases: dict of {Node, Interval}
    """

    def __init__(self, num_layers, delta, nodes, weights, biases, type="nn"):
        self.num_layers = num_layers
        self.delta = delta
        self.nodes = nodes
        self.weights = weights
        self.biases = biases
        self.type = type



class OptSolver:
    def __init__(self, dataset, inn, y_prime, x, mode=0, eps=0.0001, M=1000, x_prime=None, delta=0.0):
        self.mode = mode  # mode 0: compute counterfactual, mode 1: compute lower/upper bound of INN given a delta
        self.dataset = dataset
        self.inn = inn
        self.y_prime = y_prime  # if 0, constraint: upper output node < 0, if 1, constraint: lower output node >= 0
        self.x = x  # explainee instance x
        self.model = Model()  # initialise Gurobi optimisation model
        self.x_prime = None  # counterfactual instance
        self.eps = eps
        self.M = M
        self.delta=delta
        if x_prime is not None:
            self.x_prime = np.round(x_prime, 6)
        self.output_node_name = None

    def add_input_variable_constraints(self):
        node_var = dict()
        for feat_idx in range(self.dataset.num_features):
            # cases by feature type, add different types of variables and constraints
            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                #disc_var_list = []
                for var_idx in self.dataset.feat_var_map[feat_idx]:
                    if self.mode == 1:
                        node_var[var_idx] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                              name='x_disc_0_' + str(var_idx))
                    else:
                        node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_disc_0_' + str(var_idx))
                    #disc_var_list.append(node_var[var_idx])
                    if self.mode == 1:
                        self.model.addConstr(node_var[var_idx] == self.x_prime[var_idx],
                                             name="lbINN_disc_0_" + str(var_idx))
                self.model.update()
                #if self.mode == 0: ## remove discrete encoding constraint, applicable to binary categorical features
                ## each represented by 1 variable.
                #    self.model.addConstr(quicksum(disc_var_list) == 1, name='x_disc_0_feat' + str(feat_idx))

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                prev_var = None
                ord_var_list = []
                for i, var_idx in enumerate(self.dataset.feat_var_map[feat_idx]):
                    if self.mode == 1:
                        node_var[var_idx] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                              name='x_ord_0_' + str(var_idx))
                    else:
                        node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_ord_0_' + str(var_idx))
                    self.model.update()
                    if self.mode == 1:
                        self.model.addConstr(node_var[var_idx] == self.x_prime[var_idx],
                                             name="lbINN_disc_0_" + str(var_idx))
                    if i != 0 and self.mode == 0:
                        self.model.addConstr(prev_var >= node_var[var_idx],
                                             name='x_ord_0_var' + str(var_idx - 1) + '_geq_' + str(var_idx))
                    prev_var = node_var[var_idx]
                    ord_var_list.append(node_var[var_idx])
                if self.mode == 0:
                    self.model.addConstr(quicksum(ord_var_list) >= 1, name='x_ord_0_feat' + str(feat_idx) + '_geq1')

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                var_idx = self.dataset.feat_var_map[feat_idx][0]
                if self.mode == 1:  # no lower or upper bound limits for input variables
                    node_var[var_idx] = self.model.addVar(lb=-float('inf'), vtype=GRB.SEMICONT,
                                                          name="x_cont_0_" + str(var_idx))
                else:
                    node_var[var_idx] = self.model.addVar(lb=-float('inf'), vtype=GRB.SEMICONT,
                                                          name="x_cont_0_" + str(var_idx))
                if self.mode == 1:
                    self.model.addConstr(node_var[var_idx] == self.x_prime[var_idx],
                                         name="lbINN_disc_0_" + str(var_idx))
            self.model.update()
        return node_var

    def add_node_variables_constraints(self, node_vars, aux_vars):
        """
        create variables for nodes. Each node has the followings:
        node variable n for the final node value after ReLU,
        auxiliary variable a for the node value before ReLU.

        Constraint on each node:
        n: node variable
        a: binary variable at each node
        M: big value
        n >= 0
        n <= M(1-a)
        n <= ub(W)x + ub(B) + Ma
        n >= lb(W)x + lb(B)
        """

        for i in range(1, self.inn.num_layers):
            node_var = dict()
            aux_var = dict()
            for node in self.inn.nodes[i]:
                self.model.update()
                # hidden layers
                #w_vars = {}
                #for node1 in self.inn.nodes[i - 1]:
                #    w_var = self.model.addVar(vtype=GRB.CONTINUOUS, lb=self.inn.weights[(node1, node)].lb, ub=self.inn.weights[(node1, node)].ub, name='ww' + str(node1) + str(node))
                #    w_vars[(node1, node)] = w_var
                # Bi = Bi +- delta
                #b_var = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-float("inf"), name='b' + str(node))
                #beta_b_var = self.model.addVar(vtype=GRB.BINARY, name='beta_b_' + str(node))
                #b_var = self.model.addVar(vtype=GRB.CONTINUOUS, lb=self.inn.biases[node].lb, ub=self.inn.biases[node].ub, name='bb' + str(node))
                if i != (self.inn.num_layers - 1):
                    if self.inn.num_layers == 2:
                        continue
                    node_var[node.index] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                             name='n_' + str(node))
                    aux_var[node.index] = self.model.addVar(vtype=GRB.BINARY, name='a_' + str(node))
                    self.model.update()
                    # constraint 1: node >= 0
                    self.model.addConstr(node_var[node.index] >= 0, name="forward_pass_node_" + str(node) + "C1")
                    # constraint 2: node <= M(1-a)
                    self.model.addConstr(self.M * (1 - aux_var[node.index]) >= node_var[node.index],
                                         name="forward_pass_node_" + str(node) + "C2")
                    # constraint 3: node <= ub(W)x + ub(B) + Ma
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].ub * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].ub + self.M * aux_var[node.index] >=
                                        node_var[node.index],
                                        name="forward_pass_node_" + str(node) + "C3")
                    #self.model.addConstr(node_var[node.index]<=quicksum(w_vars[(node1, node)] * node_vars[i - 1][node1.index]
                    #                                     for node1 in self.inn.nodes[i - 1]) + b_var + self.M*aux_var[node.index])
                    #self.model.addConstr(
                    #    node_var[node.index] >= quicksum(w_vars[(node1, node)] * node_vars[i - 1][node1.index]
                    #                                     for node1 in self.inn.nodes[i - 1]) + b_var)
                    # constraint 4: node >= lb(W)x + lb(B)
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].lb * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].lb <= node_var[node.index],
                                         name="forward_pass_node_" + str(node) + "C4")
                else:
                    node_var[node.index] = self.model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                             name='n_' + str(node))
                    self.output_node_name = 'n_' + str(node)
                    # constraint 1: node <= ub(W)x + ub(B)
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].ub * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].ub >= node_var[node.index],
                                         name="output_node_pass_" + str(node) + "C1")
                    # constraint 2: node >= lb(W)x + lb(B)
                    self.model.addConstr(quicksum(
                        (self.inn.weights[(node1, node)].lb * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].lb <= node_var[node.index],
                                         name="output_node_pass_" + str(node) + "C2")
                    if self.mode == 1:
                        continue
                    # constraint3: counterfactual constraint for mode 0
                    if self.y_prime:
                        self.model.addConstr(node_var[node.index] - self.eps >= 0.0, name="output_node_lb_>=0")
                    else:
                        self.model.addConstr(node_var[node.index] + self.eps <= 0.0, name="output_node_ub_<0")
                    self.model.update()
            node_vars[i] = node_var
            if i != (self.inn.num_layers - 1):
                aux_vars[i] = aux_var
        return node_vars, aux_vars

    def create_constraints(self):
        node_vars = dict()  # dict of {layer number, {Node's idx int, Gurobi variable obj}}
        aux_vars = dict()  # dict of {layer number, {Node's idx int, Gurobi variable obj}}
        node_vars[0] = self.add_input_variable_constraints()
        node_vars, aux_vars = self.add_node_variables_constraints(node_vars, aux_vars)
        return node_vars, aux_vars

    def set_objective_l1_l0(self, node_vars):
        obj_vars_l1 = []  # each is l1 distance for 1 feature
        # obj_vars_l0 = []  # each is l0 distance for 1 feature
        for feat_idx in range(self.dataset.num_features):
            self.model.update()
            this_obj_var_l1 = self.model.addVar(vtype=GRB.SEMICONT, lb=-GRB.INFINITY, name=f"objl1_feat_{feat_idx}")
            # this_obj_var_l0 = self.model.addVar(vtype=GRB.BINARY, name=f"objl0_feat_{feat_idx}")
            var_idxs = self.dataset.feat_var_map[feat_idx]

            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                to_max = dict()
                for var_idx in var_idxs:
                    to_max[var_idx] = self.model.addVar(vtype=GRB.SEMICONT, lb=-GRB.INFINITY,
                                                        name=f"objl1_feat_disc_{var_idx}")
                    self.model.update()
                    self.model.addConstr(to_max[var_idx] >= (node_vars[0][var_idx] - self.x[var_idx]),
                                         name=f"objl1_feat_disc_{var_idx}_>=-")
                    self.model.addConstr(to_max[var_idx] >= (self.x[var_idx] - node_vars[0][var_idx]),
                                         name=f"objl1_feat_disc_{var_idx}_>=+")  # for abs()
                    self.model.update()
                self.model.addConstr(this_obj_var_l1 == max_([to_max[idx] for idx in to_max.keys()]),
                                     name=f"objl1_feat_{feat_idx}")
                # self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                #                     name=f"objl0_feat_{feat_idx}")  # if l1<= 0, l0=0

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                self.model.addConstr(this_obj_var_l1 >= (
                        quicksum([node_vars[0][idx] for idx in var_idxs]) - np.sum(self.x[var_idxs])) / (
                                             len(var_idxs) - 1), name=f"objl1_feat_{feat_idx}_>=-")
                self.model.addConstr(this_obj_var_l1 >= (
                        np.sum(self.x[var_idxs]) - quicksum([node_vars[0][idx] for idx in var_idxs])) / (
                                             len(var_idxs) - 1), name=f"objl1_feat_{feat_idx}_>=+")  # for abs()
                # self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                #                     name=f"objl0_feat_{feat_idx}")

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                self.model.addConstr(
                    this_obj_var_l1 >= (quicksum([node_vars[0][idx] for idx in var_idxs]) - np.sum(self.x[var_idxs])),
                    name=f"objl1_feat_{feat_idx}_>=-")
                self.model.addConstr(
                    this_obj_var_l1 >= (np.sum(self.x[var_idxs]) - quicksum([node_vars[0][idx] for idx in var_idxs])),
                    name=f"objl1_feat_{feat_idx}_>=+")  # for abs()
                # self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                #                     name=f"objl0_feat_{feat_idx}")

            obj_vars_l1.append(this_obj_var_l1)
            # obj_vars_l0.append(this_obj_var_l0)

        # self.model.setObjective(
        #    quicksum(obj_vars_l1) / self.dataset.num_features + quicksum(obj_vars_l0) / self.dataset.num_features,
        #    GRB.MINIMIZE)
        self.model.setObjective(quicksum(obj_vars_l1) / self.dataset.num_features, GRB.MINIMIZE)  # only L1
        self.model.update()

    def set_objective_output_node(self, node_vars):
        if self.y_prime == 1:
            self.model.setObjective(node_vars[self.inn.num_layers - 1][0], GRB.MINIMIZE)
        else:
            self.model.setObjective(node_vars[self.inn.num_layers - 1][0], GRB.MAXIMIZE)

    def compute_counterfactual(self):
        node_vars, aux_vars = self.create_constraints()
        self.set_objective_l1_l0(node_vars)  ## debug
        self.model.Params.LogToConsole = 0  # disable console output
        self.model.optimize()
        xp = []
        try:
            for v in self.model.getVars():
                if 'x_' in v.varName:
                    xp.append(v.getAttr(GRB.Attr.X))
            self.x_prime = np.array(xp)
        except:
            xp = None
        if xp is not None:
            return self.x_prime
        else:
            return None

    def compute_inn_bounds(self):
        """
        test robustness of a counterfactual
        :return: -1: infeasible, 0: not valid, 1: valid
        """
        node_vars, aux_vars = self.create_constraints()
        self.set_objective_output_node(node_vars)
        self.model.Params.LogToConsole = 0  # disable console output
        self.model.Params.NonConvex = 2
        self.model.optimize()
        res = -1
        bound = None
        try:
            bound = self.model.getVarByName(self.output_node_name).X
            res = 0
            if self.y_prime == 1:
                if bound >= 0:
                    res = 1
            if self.y_prime == 0:
                if bound < 0:
                    res = 1
            #wdict = {}
            #for v in self.model.getVars():
            #    if 'ww' in v.varName or 'bb' in v.varName:
            #        wdict[v.varName]=v.X
                    #xp.append(v.getAttr(GRB.Attr.X))
        except:
            pass
        return res, bound#, wdict


def build_dataset_feature_types(columns, ordinal, discrete, continuous):
    feature_types = dict()
    for feat in ordinal.keys():
        feature_types[columns.index(feat)] = DataType.ORDINAL
    for feat in discrete.keys():
        feature_types[columns.index(feat)] = DataType.DISCRETE
    for feat in continuous:
        feature_types[columns.index(feat)] = DataType.CONTINUOUS_REAL
    return feature_types


def build_inn_nodes(clf, num_layers):
    nodes = dict()
    for i in range(num_layers):
        this_layer_nodes = []
        if i == 0:
            num_nodes_i = clf.n_features_in_
        elif i == num_layers - 1:
            num_nodes_i = 1
        else:
            if isinstance(clf.hidden_layer_sizes, int):
                num_nodes_i = clf.hidden_layer_sizes
            else:
                num_nodes_i = clf.hidden_layer_sizes[i - 1]

        for j in range(num_nodes_i):
            this_layer_nodes.append(Node(i, j))
        nodes[i] = this_layer_nodes
    return nodes


def build_inn_weights_biases(clf, num_layers, delta, nodes):
    try:
        ws = clf.coefs_
        if clf.n_layers_ == 2:
            ws = [clf.coefs_.transpose()]
    except:
        ws = [clf.coef_.transpose()]
    try:
        bs = clf.intercepts_
        if clf.n_layers_ == 2:
            bs = [clf.intercepts_]
    except:
        bs = [clf.intercept_]
    weights = dict()
    biases = dict()
    for i in range(num_layers - 1):
        for node_from in nodes[i]:
            for node_to in nodes[i + 1]:
                w_val = round(ws[i][node_from.index][node_to.index], 8)
                weights[(node_from, node_to)] = Interval(w_val, w_val - delta, w_val + delta)
                b_val = round(bs[i][node_to.index], 8)
                biases[node_to] = Interval(b_val, b_val - delta, b_val + delta)
    return weights, biases

