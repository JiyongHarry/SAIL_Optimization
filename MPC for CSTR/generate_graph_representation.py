from pyomo.environ import *
from pyomo.dae import *
import math
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde

# #Define a global parameter for the number of points
NUM_POINTS = 50  # Descretization of the time horizon


# This is the optimization model
def get_model2(xss={}, uss={}, ucon={}, xinit=0.3, uinit=200, du_=50):
    m = ConcreteModel()
    m.V = Param(default=50)  # reactor volume [m3]
    m.k = Param(default=2)  # reaction constant [m3/mol/s]
    m.t = ContinuousSet(bounds=(0, 10))  # time [s]
    m.c = Var(m.t, bounds=(0, 1))  # concentration [mol/m3]
    m.u = Var(m.t, bounds=(0, 800))  # inlet flowe rate [m3/s]

    discretizer = TransformationFactory("dae.finite_difference")
    discretizer.apply_to(m, nfe=10, wrt=m.t, scheme="CENTRAL")
    m.c[0].fix(xinit)
    m.u[0].fix(uinit)

    m.der_u = ConstraintList()
    m.der_l = ConstraintList()
    m.ode = ConstraintList()
    for t in m.t:
        if t != 0:
            m.der_u.add(expr=(m.u[t] - m.u[t - 1]) / 0.1 <= du_)
            m.der_l.add(expr=(m.u[t] - m.u[t - 1]) / 0.1 >= -du_)
            m.ode.add(
                expr=m.c[t]
                == m.c[t - 1]
                + 0.1 * ((1 - m.c[t - 1]) * m.u[t - 1] / 50 - 2 * m.c[t - 1] ** 3)
            )
    p = {}
    time_ = [t for t in m.t]
    for t in m.t:
        k_ = list(xss.keys())
        v_ = list(xss.values())

        diff = [(t - i) ** 2 for i in xss.keys()]
        idx = np.argmin(diff)

        p[t] = v_[idx]

    def _intX(m, t):
        # return (m.c[t] - xss[math.ceil(t)])**2
        return (m.c[t] - p[t]) ** 2

    m.intX = Integral(m.t, wrt=m.t, rule=_intX)

    def _obj(m):
        return m.intX

    m.obj = Objective(rule=_obj)
    # m.obj = Objective(expr = sum( (m.c[t] - xss[math.ceil(t)])**2 for t in m.t), sense=minimize)
    return m, p


from pyomo.environ import *
import networkx as nx
from networkx.algorithms import bipartite

import matplotlib.pyplot as plt


# this function returns the graph representation of an optimization problem
def make_graphs(m, xss, du_, plot=True):
    """
    This function created the graph representation of an optimization problem
    The inputs are
        the optimization model in pyomo m
        the production target xss
        bounds on rate of change du_
        plotting option

    The output is
        B : bipartite variable constraint graph
        Bc: Constraint graph
        Bn: Variable graph
    """
    # get number of variables and constraints
    num_var = len([v[index] for v in m.component_objects(Var) for index in v])
    num_con = len([v[index] for v in m.component_objects(Constraint) for index in v])
    # create empty graph
    B = nx.Graph()
    # for every variable add a type 1 node
    for d in m.component_objects(Var):  # loop over the variables
        if d == "c":  # check if the variable is concentation
            for index in d:  # loop over indiced - time points
                # B.add_node(str(d[index]),bipartite=0)
                B.add_node(
                    str(d[index]),
                    lb=0,
                    ub=1,
                    der_lb=-1,
                    der_ub=1,
                    pr=xss[index],
                    type=1,
                )
                # here we are creating the nodes and adding features:
                # lb = lower bound
                # ub = upper bound
                # der_lb = lower bound on rate of change
                # der_ub = upper bound on rate of change
                # pr = production target at time point index
                # type = 1 if it is concentration and 0 if it is flowrate
        else:  # the variable is the flowrate
            for index in d:
                # B.add_node(str(d[index]),bipartite=0)
                B.add_node(
                    str(d[index]), lb=0, ub=800, der_lb=-du_, der_ub=du_, pr=0, type=0
                )

    from pyomo.core.expr import current as EXPR

    # for every constraint add a type 2 node
    for c in m.component_objects(Constraint):  # loop over the constraints
        for index in c:  # loop over the time indices
            B.add_node(
                str(c[index]), bipartite=1
            )  # create a node to represent the constraint
            rel_var = EXPR.identify_variables(
                c[index].body
            )  # find the variables that are present in the constraint
            for v in rel_var:
                B.add_edge(
                    str(v), str(c[index])
                )  # add an edge between variable v and constraint c[index]

    bottom_nodes, top_nodes = bipartite.sets(
        B
    )  # distinguish nodes that represent variables and constraints
    # get the variable and constraint graph:
    if len(bottom_nodes) == len(
        [v[index] for v in m.component_objects(Var) for index in v]
    ):
        Bn = bipartite.projected_graph(B, bottom_nodes)
        Bc = bipartite.projected_graph(B, top_nodes)
    else:
        Bn = bipartite.projected_graph(B, top_nodes)
        Bc = bipartite.projected_graph(B, bottom_nodes)

    if plot == True:
        print(
            "The optimization problem has {} variables and {} constraints".format(
                num_var, num_con
            )
        )
        pos = dict()
        X = bottom_nodes
        Y = top_nodes
        pos.update((n, (1, i)) for i, n in enumerate(X))  # put nodes from X at x=1
        pos.update((n, (2, i)) for i, n in enumerate(Y))  # put nodes from Y at x=2
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Variable graph")
        nx.draw(Bn, with_labels=True)
        plt.subplot(1, 3, 2)
        plt.title("Bipartite variable-constraint graph")
        nx.draw(B, with_labels=True, pos=pos)
        plt.subplot(1, 3, 3)
        plt.title("Constraint graph")
        nx.draw(Bc, with_labels=True)
        plt.show()
    return B, Bc, Bn


import torch
from torch_geometric.utils import from_networkx


def get_datapoint(Bn, label):
    # Convert NetworkX graph to PyTorch Geometric Data
    data = from_networkx(
        Bn, group_node_attrs=["lb", "ub", "der_lb", "der_ub", "pr", "type"]
    )
    # -> the above line transforms the networkX representation to Pytorch Geometric representation
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    data.y = torch.tensor(
        [label], dtype=torch.float
    )  # this is the label for the entire graph - f(x^*)
    return data


solver = SolverFactory("ipopt")
N_data_train = 1
N_data_test = 100

Nt = 11
import random

space = dde.data.GRF(T=10, kernel="RBF", length_scale=2)
feats = -space.random(1)
xs = np.linspace(0, 10, num=11)[:, None]
y = 0.5 + 0.1 * space.eval_batch(feats, xs)
xss = {}
for j in range(len(xs)):
    xss[xs[j][0]] = y[0][j]
uss = {}
x0_ = 0.5  # np.random.uniform(0,1,1)[0]
u0 = 200  # np.random.uniform(200,300,1)[0]
du_ = np.random.uniform(10, 100, 1)[0]
ucon = 100
m, p = get_model2(xss, uss, ucon, x0_, u0, du_)
# m.del_component(m.du[0])
res = solver.solve(m, tee=False)

B, Bc, Bn = make_graphs(m, xss, du_, plot=True)

label = m.obj()
data = get_datapoint(Bn, label)
print(data)
