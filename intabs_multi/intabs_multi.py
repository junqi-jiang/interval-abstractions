from gurobipy import *
import numpy as np
from intabs.intabs import DataType, UtilDataset, Interval, Node, Inn, build_dataset_feature_types


def build_inn_nodes(clf, num_layers):
    nodes = dict()
    for i in range(num_layers):
        this_layer_nodes = []
        if i == 0:
            num_nodes_i = clf.n_features_in_
        elif i == num_layers - 1:
            num_nodes_i = 3
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
        self.output_node_name = []

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
                    self.output_node_name.append('n_' + str(node))
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

    def robustness_test(self):
        node_vars, aux_vars = self.create_constraints()
        self.model.setObjective(node_vars[self.inn.num_layers - 1][self.y_prime], GRB.MINIMIZE)
        self.model.Params.LogToConsole = 0  # disable console output
        self.model.Params.NonConvex = 2
        self.model.optimize()
        bound_target = self.model.getVarByName(self.output_node_name[self.y_prime]).X

        other_labels = [i for i in [0,1,2] if i != self.y_prime]

        for other_label in other_labels:
            self.model.setObjective(node_vars[self.inn.num_layers - 1][other_label], GRB.MAXIMIZE)
            self.model.Params.LogToConsole = 0  # disable console output
            self.model.Params.NonConvex = 2
            self.model.optimize()
            bound_other = self.model.getVarByName(self.output_node_name[other_label]).X
            if bound_other >= bound_target:
                return False
        return True

