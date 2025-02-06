# Pyomo Example 2.1.3 "Updated Production plan: Product Y"

import matplotlib.pyplot as plt
import numpy as np

import shutil
import sys
import os.path

# check wheter the pyomo and cbc are installed
assert shutil.which("pyomo")
assert shutil.which("gurobi")

from pyomo.environ import *

# construct model
model = ConcreteModel()

# declare model
model.y = Var(domain=NonNegativeReals)

# declare objective function
model.profit = Objective(expr=(210 - 90 - 50 - 40) * model.y, sense=maximize)

# constraint
model.laborA = Constraint(expr=model.y <= 80)
model.laborB = Constraint(expr=model.y <= 100)

# solve the problem
SolverFactory("gurobi").solve(model).write

# display
model.profit.display()
model.y.display()
