from pyomo.environ import *
from pyomo.dae import *
import math
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import sys
import os

def get_dynamic_model(c0, u0, y, du_):
    m = ConcreteModel()
    T = np.arange(10)
    m.c = Var(T, bounds = (0,3))
    m.u = Var(T, bounds = (0,800))

    m.mass_bal = ConstraintList()
    for t in T:
        if t>=1:
            m.mass_bal.add(expr = m.c[t] == m.c[t-1] + 0.5*((1-m.c[t-1])*m.u[t-1]/50 - 2 * m.c[t-1]**3))
        else:
            m.mass_bal.add(expr = m.c[t] == c0)

    m.du_up = ConstraintList()
    m.du_lo = ConstraintList()
    for t in T:
        if t>=1:
            m.du_up.add(expr = (m.u[t]-m.u[t-1])/0.5<= du_)
            m.du_lo.add(expr = (m.u[t]-m.u[t-1])/0.5>= - du_)
    
    m.u[0].setlb(-du_*0.5+u0)
    m.u[0].setub(du_*0.5+u0)
    
    m.obj = Objective(expr = 10*sum( (m.c[t] - y[0][t])**2 for t in T), sense=minimize)
    return m,T

solver = SolverFactory('gams')
solver.options['solver'] = 'conopt'

import random
from pyomo.environ import *
import networkx as nx
from networkx.algorithms import bipartite

import matplotlib.pyplot as plt

def make_graphs(m, xss, du_):
    # print('Starting')
    # create empty graph
    num_var = len([v[index] for v in m.component_objects(Var) for index in v])
    num_con = len([v[index] for v in m.component_objects(Constraint) for index in v])
            
    B=nx.Graph()
    # for every variable add a type 1 node
    for d in m.component_objects(Var):
        if 'c' in str(d):
            # print(f'Processing {d}')
            for index in d:
                # print(f'Index = {index}')
                # B.add_node(str(d[index]),bipartite=0)
                if index==int(0):
                    # print(d[index]())
                    B.add_node(str(d[index]), lb=d[index](), ub=d[index](), der_lb=-1, der_ub=1, pr= xss[index])#, type=1)
                else:
                    B.add_node(str(d[index]), lb=0, ub=1, der_lb=-1, der_ub=1, pr= xss[index])#, type=1)
        if 'u' in str(d):
            for index in d:
                # B.add_node(str(d[index]),bipartite=0)
                B.add_node(str(d[index]),lb=d[index].lb, ub=d[index].ub, der_lb=-du_, der_ub=du_, pr= 0)#, type=0)

    from pyomo.core.expr import current as EXPR
    
    # for every constraint add a type 2 node
    for c in m.component_objects(Constraint):
        for index in c:
            rel_var=EXPR.identify_variables(c[index].body)
            for v in rel_var:
                B.add_node(str(c[index]),bipartite=1)
                B.add_edge(str(v), str(c[index]))
    # nodes_to_del =[]
    # for k in B.nodes():
    #     if B.degree(k)==0:
    #         nodes_to_del.append(k)
    
    # for k in nodes_to_del:
    #     B.remove_node(k)
    bottom_nodes, top_nodes = bipartite.sets(B)
    
    if len(bottom_nodes)==len([v[index] for v in m.component_objects(Var) for index in v]):
        Bn = bipartite.projected_graph(B,bottom_nodes)  
        Bc=bipartite.projected_graph(B,top_nodes) 
    else:
        Bn = bipartite.projected_graph(B,top_nodes)  
        Bc=bipartite.projected_graph(B,bottom_nodes)
    return B,Bc,Bn

import torch
from torch_geometric.utils import from_networkx
def get_datapoint(Bn, label):
    # Convert NetworkX graph to PyTorch Geometric Data
    data = from_networkx(Bn, group_node_attrs=["lb", "ub", "der_lb", "der_ub", "pr"])
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    data.y =  torch.tensor([label], dtype=torch.float)
    return data
# solver = SolverFactory('ipopt')
solver = SolverFactory('gams')
solver.options['solver'] = 'conopt'

data_list = []

import random
N_data= N_data # you must define this
for i in range(N_data):
    space = dde.data.GRF(T=5, kernel = 'RBF', length_scale=0.5)
    feats = -space.random(1)
    xs = np.linspace(0, 5, num=10)[:, None]
    y = 0.5 + 0.1* space.eval_batch(feats, xs)
    c0 = np.random.uniform(0.2,0.8,1)[0]
    u0 =  np.random.uniform(10,100,1)[0]
    du_ = np.random.uniform(10,40,1)[0]
    m,T = get_dynamic_model(c0,u0,y,du_)

    # solver = SolverFactory('ipopt')
    solver.solve(m,tee=False)

    # optimal solution
    xss = [k for k in y[0]]
    B,Bc,Bn = make_graphs(m, xss, du_)
    # get pytorch format
    data =  get_datapoint(Bn, m.obj())
    # data =  get_datapoint(Bn, m.u[0]())
    data_list.append(data)

torch.save(data_list, "dataset_list.pt") 

