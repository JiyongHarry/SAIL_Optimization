#new
#Pyomo Example 2.1 "Production Models ith linear constraints"

import matplotlib.pyplot as plt
import numpy as np

import shutil
import sys
import os.path

#check wheter the pyomo and cbc are installed
assert(shutil.which("pyomo"))
assert(shutil.which("cbc"))

from pyomo.environ import * #imports all the functionality from the pyomo.environ module to the script ('*' means all)

#Construct the model
model = ConcreteModel() # Represents the optimization problem. It allows you to define variables, constraints, and objectives in a structured way.

# Declare decision variable
model.x = Var(domain=NonNegativeReals) #Var: defines the decision variable for the model, #model.x: represents the number of units of product X #domain: ensures the variable is a non-negative real number(R+))

#Declare objective function
model.profit = Objective(
    expr = 40*model.x,
    sense = maximize)
#define model.profit as a Objective: specifies the goal of the optimization problem. #expr: write equation after expr =, #sense: maximize or minimize


#Declare constraints
model.demand = Constraint(expr = model.x <= 40)
model.laborA = Constraint(expr = model.x <= 80)
model.laborB = Constraint(expr = 2*model.x <= 100)

#Solve the problem
SolverFactory('cbc').solve(model).write() #inside the salverfactory, different solver can be used such as 'cbc', 'gurobi', 'ipopt'

#Display the results
model.profit.display()
model.x.display()

#In formatted forms
print(f"Profit = {model.profit()} per week")
print(f"X = {model.x()} units per week")
