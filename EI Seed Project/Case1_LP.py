# Basic imports
from pyomo.environ import *
import pandas as pd


# ----------------------------------------------------
# Load bus data from CSV
# ----------------------------------------------------
df_bus = pd.read_csv("EI Seed Project/TEXAS 123-BT params (bus).csv")
df_gen = pd.read_csv("EI Seed Project/TEXAS 123-BT params (generation).csv")
df_line = pd.read_csv("EI Seed Project/TEXAS 123-BT params (line).csv")

# ----------------------------------------------------
# Model
# ----------------------------------------------------
model = ConcreteModel()

# ----------------------------------------------------
# Sets
# ----------------------------------------------------
# Global parameters for time horizons
time_yearsHorizon = 1  # Number of years in the planning horizon
time_daysHorizon = 2  # Number of days in the planning horizon
time_hoursHorizon = 2  # Number of hours in a day
BaseMVA = 100  # Base power in MVA, used for per unit calculations

# Create a set of valid (from_bus, to_bus) pairs
valid_line_pairs = sorted(
    (row["From Bus Number"], row["To Bus Number"]) for _, row in df_line.iterrows()
)
# Create a set of valid (bus_number, fuel_type) pairs
valid_gen_paris = sorted(
    (row["Bus Number"], row["Fuel type"]) for _, row in df_gen.iterrows()
)

model.N = Set(initialize=df_bus["Bus Number"].unique())  # Set of buses
model.G = Set(
    initialize=valid_gen_paris, dimen=2
)  # Set of generators (as pairs of bus number and fuel type)
model.L = Set(
    initialize=valid_line_pairs, dimen=2
)  # Set of lines (as pairs of bus numbers)
model.D = Set(initialize=range(1, time_daysHorizon + 1))  # Set of days
model.H = Set(initialize=range(1, time_hoursHorizon + 1))  # Set of hours
model.I_gen = Set(
    initialize=df_gen["Fuel type"].unique()
)  # Set of generator technologies
model.I_gen_TH = Set(
    initialize=df_gen[
        df_gen["Fuel type"].isin(["Natural Gas", "Coal", "Nuclear", "Hydro"])
    ]["Fuel type"].unique()
)  # Set of thermal generator technologies
# Hydro is included as a parameter from the reference, but should not be considered as investment variable.
model.I_gen_RN = Set(
    initialize=df_gen[df_gen["Fuel type"].isin(["Wind", "Solar"])]["Fuel type"].unique()
)  # Set of renewable generators
model.W = Set(initialize=df_bus["Weather Zone"].unique())  # Set of weather zones
model.C = Set(
    initialize=df_bus["Weather Zone"].unique()
)  # Regional sets of data centers load zone (TBD)
model.M = Set(
    initialize=df_bus["Weather Zone"].unique()
)  # Regional sets of chemical manufacturing load zone (TBD)

# ----------------------------------------------------
# Parameters
# ----------------------------------------------------
# ----- Bus Parameters ----
bus_name_dict = {row["Bus Number"]: row["Bus Name"] for _, row in df_bus.iterrows()}
bus_latitude_dict = {
    row["Bus Number"]: row["Bus latitude"] for _, row in df_bus.iterrows()
}
bus_longitude_dict = {
    row["Bus Number"]: row["Bus longitude"] for _, row in df_bus.iterrows()
}
bus_genBool_dict = {
    row["Bus Number"]: row["Gen bus/ Non-gen bus"] for _, row in df_bus.iterrows()
}
bus_nominalVoltage_dict = {
    row["Bus Number"]: row["Nominal Voltage (KV)"] for _, row in df_bus.iterrows()
}
bus_weatherZone_dict = {
    row["Bus Number"]: row["Weather Zone"] for _, row in df_bus.iterrows()
}
bus_dataCenterZone_dict = {
    row["Bus Number"]: row["Weather Zone"] for _, row in df_bus.iterrows()
}  # SHOULD BE MODIFIED LATER
bus_chemicalManuZone_dict = {
    row["Bus Number"]: row["Weather Zone"] for _, row in df_bus.iterrows()
}  # SHOULD BE MODIFIED LATER

model.bus_name = Param(model.N, initialize=bus_name_dict, within=Any)
model.bus_latitude = Param(model.N, initialize=bus_latitude_dict)
model.bus_longitude = Param(model.N, initialize=bus_longitude_dict)
model.bus_genBool = Param(model.N, initialize=bus_genBool_dict)
model.bus_nominalVoltage = Param(model.N, initialize=bus_nominalVoltage_dict)
model.bus_weatherZone = Param(
    model.N, initialize=bus_weatherZone_dict, within=Any
)  # within=Any: this allows for any type of data, including strings or integers. default is Real, which is a floating-point number.
model.dataCenterZone = Param(model.N, initialize=bus_dataCenterZone_dict, within=Any)
model.chemicalManuZone = Param(
    model.N, initialize=bus_chemicalManuZone_dict, within=Any
)


# ----- Generator Parameters ----
# Generator initial capacity based on the bus number and fuel type
gen_c_gen_init_dict = (
    df_gen.groupby(["Bus Number", "Fuel type"])["Pmax (MW)"].sum().to_dict()
)  # Group by (Bus Number, Fuel type) and sum Pmax (MW)

# Weighted average parameters based on the capacity of each generator type: costs and ramping rates
gen_c0_dict = {}
gen_c1_dict = {}
gen_csu_dict = {}
gen_r_ramp_dict = {}

grouped = df_gen.groupby(["Bus Number", "Fuel type"])
for (bus, fuel), group in grouped:
    pmax_sum = group["Pmax (MW)"].sum()
    if pmax_sum == 0:
        gen_c0_dict[(bus, fuel)] = 0
        gen_c1_dict[(bus, fuel)] = 0
        gen_csu_dict[(bus, fuel)] = 0
        gen_r_ramp_dict[(bus, fuel)] = 0

    else:
        gen_c0_dict[(bus, fuel)] = (
            group["C0($/MWh)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()
        gen_c1_dict[(bus, fuel)] = (
            group["C1($/MWh)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()
        gen_csu_dict[(bus, fuel)] = (
            group["Csu($)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()
        gen_r_ramp_dict[(bus, fuel)] = (
            group["Ramping Rate(MW/min)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()


model.gen_c_gen_init = Param(model.G, initialize=gen_c_gen_init_dict)
model.gen_c0 = Param(model.G, initialize=gen_c0_dict)
model.gen_c1 = Param(model.G, initialize=gen_c1_dict)
model.gen_csu = Param(model.G, initialize=gen_csu_dict)
model.gen_r_ramp = Param(model.G, initialize=gen_r_ramp_dict)

# ----- Line Parameters ----
line_r_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["R, pu"]
    for _, row in df_line.iterrows()
}
line_x_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["X, pu"]
    for _, row in df_line.iterrows()
}
line_b_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["B, pu"]
    for _, row in df_line.iterrows()
}
line_c_trans_init_dict = (
    df_line.groupby(["From Bus Number", "To Bus Number"])["Capacity (MW)"]
    .sum()
    .to_dict()
)  # line capacity of n to n_prime should be sum of multiple lines of n to n_prime
line_mile_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["Length (Mile)"]
    for _, row in df_line.iterrows()
}

model.line_r = Param(model.L, initialize=line_r_dict)
model.line_x = Param(model.L, initialize=line_x_dict)
model.line_b = Param(model.L, initialize=line_b_dict)
model.line_c_trans_init = Param(model.L, initialize=line_c_trans_init_dict)
model.line_mile = Param(model.L, initialize=line_mile_dict)

# ----- Storage Parameters ----

# ----- Load Parameters ----
# Load parameters for data centers and chemical manufacturing
model.D_res = Param(model.N, model.D, model.H, initialize=0)
model.D_dataCenter = Param(model.C, model.D, model.H, initialize=0)
model.D_chemManu = Param(
    model.M, model.D, model.H, initialize=0
)  # Chemical manufacturing load

# ----------------------------------------------------
# Variables
# ----------------------------------------------------
# Example variable: x[n,d,h]
# model.x = Var(model.N, model.D, model.H, domain=NonNegativeReals)

# model.p_gen = Var(
#     model.N, model.I_gen, model.D, model.H, domain=NonNegativeReals
# )  # Power generated by each generator at each bus, day, and hour

model.p_gen = Var(
    model.G, model.D, model.H, domain=NonNegativeReals
)  # Power generated by each generator at each bus, day, and hour
model.p_discharge = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power discharged by each generator at each bus, day, and hour
model.p_charge = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power charged by each generator at each bus, day, and hour
model.theta = Var(
    model.N, model.D, model.H, domain=Reals
)  # Phase angle at each bus n, day, and hour
model.p_chemManu = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power consumed by chemical manufacturing at each region, day, and hour
model.p_dataCenter = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power consumed by data centers at each region, day, and hour
model.curt = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power curtailed at each bus, day, and hour


# ----------------------------------------------------
# Constraints
# ----------------------------------------------------
# operational constraints
def const_oper_energyBalance_rule(m, n, d, h):
    p_gen_sum = sum(
        m.p_gen[g, d, h] for g in m.G if g[0] == n
    )  # Sum of power by generators at bus n

    p_transmission_out_sum = sum(
        BaseMVA / m.line_x[n, k] * (m.theta[n, d, h] - m.theta[k, d, h])
        for (from_bus, k) in m.L
        if from_bus == n
    )  # Sum of power transmitted out from bus n

    p_transmission_in_sum = sum(
        BaseMVA / m.line_x[k, n] * (m.theta[k, d, h] - m.theta[n, d, h])
        for (k, to_bus) in m.L
        if to_bus == n
    )  # Sum of power transmitted into bus n

    return (
        p_gen_sum
        + m.p_discharge[n, d, h]
        - m.p_charge[n, d, h]
        + p_transmission_in_sum
        - p_transmission_out_sum
        == m.D_res[n, d, h]
        + m.p_chemManu[n, d, h]
        + m.p_dataCenter[n, d, h]
        - m.curt[n, d, h]
    )


model.const_oper_energyBalance = Constraint(
    model.N, model.D, model.H, rule=const_oper_energyBalance_rule
)
# print(model.const_oper_energyBalance[100, 1, 1].expr)


def const_oper_loadBalanceOfDataCenter_rule(m, c, d, h):
    return (
        sum(m.p_dataCenter[n, d, h] for n in m.N if m.dataCenterZone[n] == c)
        == m.D_dataCenter[c, d, h]
    )


model.const_oper_loadBalanceOfDataCenter = Constraint(
    model.C, model.D, model.H, rule=const_oper_loadBalanceOfDataCenter_rule
)

print(model.const_oper_loadBalanceOfDataCenter["WEST", 1, 1].expr)
# print(model.const_oper_energyBalance[100, 1, 1].expr)


def const_oper_loadBalanceOfChemManu_rule(m, m_idx, d, h):
    return (
        sum(m.p_chemManu[n, d, h] for n in m.N_m[m_idx]) == m.D_chemManu[m_idx, d, h]
    )  # Not to confuse m_index as a set of M with model m


def const_oper_genCapacity_rule(m, n, i, d, h):
    return m.p_gen[n, i, d, h] <= m.c_gen0[n, i] + m.c_gen[n, i]


def const_oper_genRampingUp_rule(m, n, i, d, h):
    if (d == 1) and (h == 1):
        return Constraint.Skip
    elif h == 1:
        return m.p_gen[n, i, d, h] - m.p_gen[n, i, d - 1, m.H.last()] <= m.R_ramp[i]
    else:
        return m.p_gen[n, i, d, h] - m.p_gen[n, i, d, h - 1] <= m.R_ramp[i]


def const_oper_genRampingDown_rule(m, n, i, d, h):
    if (d == 1) and (h == 1):
        return Constraint.Skip
    elif h == 1:
        return m.p_gen[n, i, d, h] - m.p_gen[n, i, d - 1, m.H.last()] >= -m.R_ramp[i]
    else:
        return m.p_gen[n, i, d, h] - m.p_gen[n, i, d, h - 1] >= -m.R_ramp[i]


# transimission
def const_oper_transCapacityUpper_rule(m, n, n_prime, d, h):
    if n >= n_prime:
        return Constraint.Skip
    return (
        m.B[n, n_prime] * (m.theta[n, d, h] - m.theta[n_prime, d, h])
        <= m.c_trans0[n, n_prime] + m.c_trans[n, n_prime]
    )


def const_oper_transCapacityLower_rule(m, n, n_prime, d, h):
    if n >= n_prime:
        return Constraint.Skip
    return m.B[n, n_prime] * (m.theta[n, d, h] - m.theta[n_prime, d, h]) >= -(
        m.c_trans0[n, n_prime] + m.c_trans[n, n_prime]
    )


# storage
def const_stor_storageLevel_rule(m, n, d, h):
    if (d == 1) and (h == 1):  # initial level should be designated
        return (
            m.p_level[n, d, h]
            == m.p_level0[n, d]
            + m.eta_charge * m.p_charge[n, d, h]
            + m.eta_discharge * m.p_discharge[n, d, h]
        )
    else:
        return (
            m.p_level[n, d, h]
            == m.p_level[n, d, h - 1]
            + m.eta_charge * m.p_charge[n, d, h]
            + m.eta_discharge * m.p_discharge[n, d, h]
        )


def const_stor_storageLevelRollOver_rule(m, n, d):
    if d == 1:
        return Constraint.Skip
    else:
        return m.p_level[n, d, 1] == m.p_level[n, d - 1, m.H.last()]


def const_stor_storageCapacity_rule(m, n, d, h):
    return m.p_level[n, d, h] <= m.c_stor0[n] + m.c_stor[n]


# investment
def const_invest_genCapacity_rule(m, n, i):
    return (
        m.c_gen[n, i] <= m.c_gen_max[n, i]
    )  # Maximum capacity for each generator type at each bus


def const_invest_transCapacity_rule(m, n, n_prime):
    return (
        m.c_trans[n, n_prime] <= m.c_trans_max[n, n_prime]
    )  # Maximum capacity for each transmission line between buses


def const_invest_storCapacity_rule(m, n):
    return (
        m.c_stor[n] <= m.c_stor_max[n]
    )  # Maximum capacity for each storage at each bus


model.const_oper_energyBalance = Constraint(
    model.N, model.D, model.H, rule=const_oper_energyBalance_rule
)
model.const_oper_loadBalanceOfDataCenter = Constraint(
    model.C, model.D, model.H, rule=const_oper_loadBalanceOfDataCenter_rule
)
model.const_oper_loadBalanceOfChemManu = Constraint(
    model.M, model.D, model.H, rule=const_oper_loadBalanceOfChemManu_rule
)
model.const_oper_genCapacity = Constraint(
    model.N, model.I_gen, model.D, model.H, rule=const_oper_genCapacity_rule
)
model.const_oper_genRampingUp = Constraint(
    model.N, model.I_gen_TH, model.D, model.H, rule=const_oper_genRampingUp_rule
)
model.const_oper_genRampingDown = Constraint(
    model.N, model.I_gen_TH, model.D, model.H, rule=const_oper_genRampingDown_rule
)
model.const_oper_storageLevel = Constraint(
    model.N, model.D, model.H, rule=const_stor_storageLevel_rule
)
model.const_oper_storageLevelRollOver = Constraint(
    model.N, model.D, rule=const_stor_storageLevelRollOver_rule
)
model.const_oper_storageCapacity = Constraint(
    model.N, model.D, model.H, rule=const_stor_storageCapacity_rule
)
model.const_oper_transCapacity_upper = Constraint(
    model.N, model.N, model.D, model.H, rule=const_oper_transCapacityUpper_rule
)
model.const_oper_transCapacity_lower = Constraint(
    model.N, model.N, model.D, model.H, rule=const_oper_transCapacityLower_rule
)

# model.pprint # for checking the infos.


# ----------------------------------------------------
# Objective
# ----------------------------------------------------
def obj_rule(m):
    # Placeholder: sum over all variables (customize as needed)
    return sum(m.x[n, d, h] for n in m.N for d in m.D for h in m.H)

    # model.Objective = Objective(rule=obj_rule, sense=maximize)

    # --- Constraints ---
    # def example_constraint_rule(m, n, d, h):
    # Placeholder: x[n,d,h] <= D_res[n,d,h]
    return m.x[n, d, h] <= m.D_res[n, d, h]


# model.ExampleConstraint = Constraint(
#     model.N, model.D, model.H, rule=example_constraint_rule
# )

# --- Notes ---
# - Add more sets, parameters, variables, objectives, and constraints as needed.
# - For each parameter, load from Excel and convert to a dictionary keyed by the set indices.
