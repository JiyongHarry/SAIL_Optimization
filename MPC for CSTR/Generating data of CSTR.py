from pyomo.environ import *
from pyomo.dae import *
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import time

# Define a global parameter for the number of points
NUM_POINTS = 50  # Descretization of the time horizon
TotalCase = 1  # Number of cases

# Record the start time
start_time = time.time()


def get_model_variable_volume(xss={}, uss={}, ucon={}, xinit=0.3, uinit=200):
    # The dynamic model is
    # dc/dt = (cin - c(t)) u(t) / V - 2 * c(t)^3
    # c is the concentration, u is the inlet, V the volume (constant), 2 is the reaction constant
    # reaction is 3rd order 3A->B
    m = ConcreteModel()
    m.V = Param(default=50)  # reactor volume [m3]
    m.k = Param(default=2)  # reaction constant [m3/mol/s]
    m.t = ContinuousSet(bounds=(0, 10))  # time [s]
    m.c = Var(m.t, bounds=(0, 1))  # concentration [mol/m3]
    m.u = Var(m.t, bounds=(0, 800))  # inlet flowe rate [m3/s]

    m.dc = DerivativeVar(m.c, wrt=m.t)  # derivative dc/dt
    m.du = DerivativeVar(m.u, wrt=m.t, bounds=(None, None))  # derivative du/dt

    m.ode = Constraint(
        m.t,
        rule=lambda m, t: m.dc[t] == (1 - m.c[t]) * m.u[t] / m.V - m.k * m.c[t] ** 3,
    )
    ## Q: WHAT IS THE RULE OF LAMDA? / A: when it comes to the set, need to call lamda m for model, and the set

    # discretize differential equations
    discretizer = TransformationFactory("dae.finite_difference")
    discretizer.apply_to(m, nfe=NUM_POINTS, wrt=m.t, scheme="BACKWARD")
    # m.ode.pprint()

    # m.c[0].fix(xinit)
    # m.u[0].fix(uinit)

    # limits on how fast the flowrates can change
    m.der_u = Constraint(m.t, rule=lambda m, t: m.du[t] <= 20)
    m.der_l = Constraint(m.t, rule=lambda m, t: m.du[t] >= -20)

    ## Q: HOW AND WHAT'S THE PURPOSE OF THE FOLLOWING LINES?
    # the following code is for the mismatch between the model time line and the prduction target time line.
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


import deepxde as dde
import random

# Initialize 2D arrays to store p and m.u
p_val_array = np.empty((TotalCase, NUM_POINTS + 1))
u_val_array = np.empty((TotalCase, NUM_POINTS + 1))

for case_num in range(TotalCase):  # generate cases
    print(f"case_num: {case_num}")
    # Generate a random production target
    space = dde.data.GRF(
        T=10, kernel="RBF", length_scale=2
    )  # creat Gaussian random field with time horizon of 10 and RBF kernel, lenght scale of 2
    feats = -space.random(1)
    xs = np.linspace(0, 10, num=NUM_POINTS + 1)[
        :, None
    ]  # - time. make column vector of (NUM_POINTS + 1, 1)

    y = 0.5 + 0.1 * space.eval_batch(
        feats, xs
    )  # - production target ## Q: WHY SCALE AND SHIFT? // A: TO MAKE C to locate from 0 to 1
    xss = {}
    for j in range(len(xs)):
        xss[xs[j][0]] = y[0][j]
    uss = {}
    x0_ = 0.2  # np.random.uniform(0,1,1)[0]
    u0 = 250  # np.random.uniform(200,1500,1)[0]
    ucon = 100
    m, p = get_model_variable_volume(
        xss, uss, ucon, x0_, u0
    )  # Q: WHY DO WE USE USS, UCO, X0, U0 IF THEY ARE NOT USED?

    solver = SolverFactory("ipopt")
    results = solver.solve(m, tee=False)
    # print(results)

    # checking status of the problem differentiate each of the results in optimal, infeasible

    if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal
    ):
        print("Solver Status:", results.solver.status)
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("Solver Status:", results.solver.status)
        print("!!!Exception detected!!!")
        break
    else:
        print("Solver Status:", results.solver.status)
        print("!!!Exception detected!!!")
        break

    # Store the results
    t_ = [t for t in m.t]
    uin_sol = [m.u[t]() for t in m.t]
    p_sol = [p[t] for t in m.t]

    for j, (p_val, u_val) in enumerate(zip(p_sol, uin_sol)):
        p_val_array[case_num, j] = p_val
        u_val_array[case_num, j] = u_val

# Record the end time
end_time = time.time()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"p_val_array: {p_val_array}")
print(f"u_val_array: {u_val_array}")

# Save 2D arrays to separate sheets in an Excel file
with pd.ExcelWriter(
    f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_{TotalCase}_data_CSTR.xlsx"
) as writer:
    pd.DataFrame(p_val_array).to_excel(writer, sheet_name="p_val", index=False)
    pd.DataFrame(u_val_array).to_excel(writer, sheet_name="u_val", index=False)


# Function to read p_val and u_val from the Excel file
def read_data(file_path):
    with pd.ExcelFile(file_path) as reader:
        p_val_df = pd.read_excel(reader, sheet_name="p_val")
        u_val_df = pd.read_excel(reader, sheet_name="u_val")
    p_val_array = p_val_df.to_numpy()
    u_val_array = u_val_df.to_numpy()
    return p_val_array, u_val_array
