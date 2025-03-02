from pyomo.environ import *
from pyomo.dae import *
import math
import matplotlib.pyplot as plt
import numpy as np


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

# generate a random production target

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
m, p = get_model_variable_volume(xss, uss, ucon, x0_, u0)
# Q: WHY DO WE USE USS, UCO, X0, U0 IF THEY ARE NOT USED?

m.pprint()

plt.title("Production target")
plt.plot([t for t in m.t], p.values())
plt.xlabel("Time")
plt.ylabel("Concentration")

solver = SolverFactory(
    "ipopt"
)  # Q: HOW DOES THE MODEL KNOW WHICH ARE DECITION VARIABLES AND WHICH ARE PARAMETERS?
res = solver.solve(m, tee=True)
# store the results
t_ = [t for t in m.t]
uin_sol = [m.u[t]() for t in m.t]
c_sol = [m.c[t]() for t in m.t]

# Get the type and contents of p
print("Type of p:", type(p))
print("Contents of p:", p)

# Get the type and contents of uin_sol
print("Type of uin_sol:", type(uin_sol))
print("Contents of uin_sol:", uin_sol)

plt.plot(t_, uin_sol)
plt.title("Inlet Flow Rate Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Inlet Flow Rate (m^3/s)")
plt.show()
