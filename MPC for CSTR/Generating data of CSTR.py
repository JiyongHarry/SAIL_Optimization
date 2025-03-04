from pyomo.environ import *
from pyomo.dae import *
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    ## Q: WHAT IS THE RULE OF LAMDA?

    # discretize differential equations
    discretizer = TransformationFactory("dae.finite_difference")
    discretizer.apply_to(m, nfe=50, wrt=m.t, scheme="BACKWARD")

    # m.c[0].fix(xinit)
    # m.u[0].fix(uinit)

    # limits on how fast the flowrates can change
    m.der_u = Constraint(m.t, rule=lambda m, t: m.du[t] <= 20)
    m.der_l = Constraint(m.t, rule=lambda m, t: m.du[t] >= -20)

    ## Q: HOW AND WHAT'S THE PURPOSE OF THE FOLLOWING LINES?
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

    m.obj = Objective(rule=_obj)  # Q: WHAT DOES RULE=_0BJ MEAN?
    # m.obj = Objective(expr = sum( (m.c[t] - xss[math.ceil(t)])**2 for t in m.t), sense=minimize)
    return m, p


import deepxde as dde
import random

# Initialize lists to store the results
all_data = []

# Generate 100 cases
for case_num in range(100):
    # Generate a random production target
    space = dde.data.GRF(
        T=10, kernel="RBF", length_scale=2
    )  # creat Gaussian random field with time horizon of 10 and RBF kernel, lenght scale of 2
    feats = -space.random(1)
    xs = np.linspace(0, 10, num=51)[:, None]  # - time. make column vector of (51,1)
    y = 0.5 + 0.1 * space.eval_batch(
        feats, xs
    )  # - production target ## Q: WHY SCALE AND SHIFT?
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
    res = solver.solve(m, tee=False)

    # Store the results
    t_ = [t for t in m.t]
    uin_sol = [m.u[t]() for t in m.t]
    p_sol = [p[t] for t in m.t]

    for t, p_val, u_val in zip(t_, p_sol, uin_sol):
        all_data.append([case_num, t, p_val, u_val])

# Convert list to DataFrame
df = pd.DataFrame(
    all_data, columns=["case", "time", "production_target", "inlet_flow_rate"]
)

# Save to CSV
df.to_csv(
    "/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_data_CSTR.csv",
    index=False,
)

# Plot production target over time for all cases
plt.figure(figsize=(10, 6))
for case_num in range(100):
    case_data = df[df["case"] == case_num]
    plt.plot(
        case_data["time"], case_data["production_target"], label=f"Case {case_num}"
    )

plt.title("Production Target Over Time for All Cases")
plt.xlabel("Time (s)")
plt.ylabel("Production Target (Concentration)")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.show()

# Plot inlet flow rate over time for all cases
plt.figure(figsize=(10, 6))
for case_num in range(100):
    case_data = df[df["case"] == case_num]
    plt.plot(case_data["time"], case_data["inlet_flow_rate"], label=f"Case {case_num}")

plt.title("Inlet Flow Rate Over Time for All Cases")
plt.xlabel("Time (s)")
plt.ylabel("Inlet Flow Rate (m^3/s)")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.show()

# # First figure: Production Target
# fig1 = plt.figure()  # Explicitly create a new figure
# plt.plot([t for t in m.t], p.values())
# plt.title("Production Target")
# plt.xlabel("Time")
# plt.ylabel("Concentration")

# # Second figure: Inlet Flow Rate
# fig2 = plt.figure()  # Explicitly create another new figure
# plt.plot(t_, uin_sol)
# plt.title("Inlet Flow Rate Over Time")
# plt.xlabel("Time (s)")
# plt.ylabel("Inlet Flow Rate (m^3/s)")
# plt.show()
