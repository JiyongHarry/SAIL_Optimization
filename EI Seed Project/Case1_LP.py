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

# Define sets
model.N = Set(initialize=df_bus["Bus Number"].unique())  # Set of buses
model.G = Set(
    initialize=valid_gen_paris, dimen=2
)  # Set of generators (as pairs of bus number and fuel type. use tuples)
model.L = Set(
    initialize=valid_line_pairs, dimen=2
)  # Set of lines (as pairs of bus numbers. use tuples)
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
}  # TBD
bus_chemicalManuZone_dict = {
    row["Bus Number"]: row["Weather Zone"] for _, row in df_bus.iterrows()
}  # TBD

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
)  # Group by (Bus Number, Fuel type) and sum Pmax (MW) # Note that there is multiple generators of same fuel type at the same bus

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
        ).sum()  # CAPEX cost from TX-123BT
        gen_c1_dict[(bus, fuel)] = (
            group["C1($/MWh)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()  # CAPEX cost from TX-123BT
        gen_csu_dict[(bus, fuel)] = (
            group["Csu($)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()  # start-up cost from TX-123BT
        gen_r_ramp_dict[(bus, fuel)] = (
            group["Ramping Rate(MW/min)"] * group["Pmax (MW)"] / pmax_sum
        ).sum()  # ramping rate from TX-123BT


model.gen_c_gen_init = Param(model.G, initialize=gen_c_gen_init_dict)
model.gen_c0 = Param(model.G, initialize=gen_c0_dict)
model.gen_c1 = Param(model.G, initialize=gen_c1_dict)
model.gen_csu = Param(model.G, initialize=gen_csu_dict)
model.gen_r_ramp = Param(model.G, initialize=gen_r_ramp_dict)

# ----- Line Parameters ----
line_r_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["R, pu"]
    for _, row in df_line.iterrows()
}  # line resistance in pu
line_x_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["X, pu"]
    for _, row in df_line.iterrows()
}  # line reactance in pu
line_b_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["B, pu"]
    for _, row in df_line.iterrows()
}  # line susceptance in pu
line_c_trans_init_dict = (
    df_line.groupby(["From Bus Number", "To Bus Number"])["Capacity (MW)"]
    .sum()
    .to_dict()
)  # capacity of the line connecting bus n to n_prime should be sum of multiple lines of n to n_prime
line_mile_dict = {
    (row["From Bus Number"], row["To Bus Number"]): row["Length (Mile)"]
    for _, row in df_line.iterrows()
}  # length of the line connecting bus n to n_prime in miles (used for CAPEX calculation)

model.line_r = Param(model.L, initialize=line_r_dict)
model.line_x = Param(model.L, initialize=line_x_dict)
model.line_b = Param(model.L, initialize=line_b_dict)
model.line_c_trans_init = Param(model.L, initialize=line_c_trans_init_dict)
model.line_mile = Param(model.L, initialize=line_mile_dict)


# ----- Storage Parameters ----
model.stor_p_stor_level_init = Param(model.N, initialize=0.1)  # TBD
model.eta_charge = Param(initialize=0.5)  # TBD
model.eta_discharge = Param(initialize=0.5)  # TBD
model.c_stor_init = Param(model.N, initialize=0)  # TBD


# ----- Load Parameters ----
# Load parameters for data centers and chemical manufacturing
model.D_res = Param(
    model.N, model.D, model.H, initialize=0
)  # TBD (by making representative days of residential load from TEXAS-123BT)
model.D_dataCenter = Param(
    model.C, model.D, model.H, initialize=0
)  # TBD (by regional data center load data from other sources)
model.D_chemManu = Param(
    model.M, model.D, model.H, initialize=0
)  # TBD (by regional chemical manufacturing load data from other sources)

# ----- Investment Parameters ----
# Maximum capacity for generators, transmission lines, and storage
model.c_gen_max = Param(
    model.G, initialize=10000
)  # Maximum gen capacity (MW) for generators. TBD
model.c_trans_max = Param(
    model.L, initialize=5000
)  # Maximum capacity for transmission lines (MW). TBD
model.c_stor_max = Param(
    model.N, initialize=500
)  # Maximum capacity for storage at each bus (MW). TBD


# ----- Objective Function Parameters ----
# Cost parameters for CAPEX and OPEX
# ENTIRE VALUES SHOULD BE REPLACED (TBD)
# CAREFUL ABOUT UNITS WHEN IMPLOYING A NEW PARAMETER eg. $/MW, $/MWh

obj_gamma_gen_dict = {
    "Natural Gas": 1200,
    "Coal": 1500,
    "Nuclear": 3000,
    "Wind": 900,
    "Solar": 800,
    "Hydro": 2000,
}  # CAPEX coefficients: $/MW for each generator type
obj_gamma_trans_dict = 1.343800e6
# CAPEX coefficient: Constant unless the system makes new transmission lines. (capacity increase will NOT affect the CAPEX) $/mile for each line (SOURCE: Transmission Capital Cost $permile (Estimation of Transmission Costs for New Generation, UT) )
obj_gamma_stor_dict = 800  # CAPEX coefficient: $/MW for storage
obj_phi_dict = 1  # Discount factor
obj_pi_gen_dict = {
    "Natural Gas": 25,
    "Coal": 20,
    "Nuclear": 15,
    "Wind": 5,
    "Solar": 4,
    "Hydro": 8,
}  # OPEX coefficients: $/MW for each generator type
obj_pi_trans_dict = {
    (n, n_prime): 20 for (n, n_prime) in model.L
}  # $/MWh for each line
obj_pi_stor_dict = 10  # $/MWh for storage
obj_pi_curt_dict = 5000  # $/MWh for curtailment. Note for the unit MWh may need integration over time (SOURCE: Grossman)

model.obj_gamma_gen = Param(model.I_gen, initialize=obj_gamma_gen_dict)
model.obj_gamma_trans = Param(initialize=obj_gamma_trans_dict)
model.obj_gamma_stor = Param(initialize=obj_gamma_stor_dict)
model.obj_phi = Param(initialize=obj_phi_dict)
model.obj_pi_gen = Param(model.I_gen, initialize=obj_pi_gen_dict)
model.obj_pi_trans = Param(model.L, initialize=obj_pi_trans_dict)
model.obj_pi_stor = Param(initialize=obj_pi_stor_dict)
model.obj_pi_curt = Param(initialize=obj_pi_curt_dict)


# ----------------------------------------------------
# Variables
# ----------------------------------------------------
model.p_gen = Var(
    model.G, model.D, model.H, domain=NonNegativeReals
)  # Power generated by each generator (n, i) at day (d) and hour (h)
model.p_stor_discharge = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power discharged by storage at each bus (n), day (d), and hour (h)
model.p_stor_charge = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power charged by storage at each bus (n), day (d), and hour (h)
model.theta = Var(
    model.N, model.D, model.H, domain=Reals
)  # Phase angle at each bus (n), day (d), and hour (h). Note: We may consider different line set here for the theta if we use TX-123BT
model.p_load_chemManu = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power consumed by chemical manufacturing at each bus (n), day (d), and hour (h)
model.p_load_dataCenter = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power consumed by data centers at at each bus (n), day (d), and hour (h)
model.curt = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power curtailed at at each bus (n), day (d), and hour (h)
model.p_stor_level = Var(
    model.N, model.D, model.H, domain=NonNegativeReals
)  # Power of storage level at each bus (n), day (d), and hour (h)
model.c_gen = Var(model.G, domain=NonNegativeReals)  # capacity of each generator (n, i)
model.c_trans = Var(
    model.L, domain=NonNegativeReals
)  # capacity of transmission lines (n, n_prime)
model.c_stor = Var(
    model.N, domain=NonNegativeReals
)  # capacity of storage at each bus (n)


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
        + m.p_stor_discharge[n, d, h]
        - m.p_stor_charge[n, d, h]
        + p_transmission_in_sum
        - p_transmission_out_sum
        == m.D_res[n, d, h]
        + m.p_load_chemManu[n, d, h]
        + m.p_load_dataCenter[n, d, h]
        - m.curt[n, d, h]
    )  # Power balance should be modified based on which transmission model is used. TBD


model.const_oper_energyBalance = Constraint(
    model.N, model.D, model.H, rule=const_oper_energyBalance_rule
)
# print(model.const_oper_energyBalance[100, 1, 1].expr)


def const_oper_loadBalanceOfDataCenter_rule(m, c, d, h):
    return (
        sum(m.p_load_dataCenter[n, d, h] for n in m.N if m.dataCenterZone[n] == c)
        == m.D_dataCenter[c, d, h]
    )  # regional load balance for data centers


model.const_oper_loadBalanceOfDataCenter = Constraint(
    model.C, model.D, model.H, rule=const_oper_loadBalanceOfDataCenter_rule
)


def const_oper_loadBalanceOfChemManu_rule(m, m_idx, d, h):
    return (
        sum(m.p_load_chemManu[n, d, h] for n in m.N if m.chemicalManuZone[n] == m_idx)
        == m.D_chemManu[m_idx, d, h]
    )  # regional load balance for data centers # Not to confuse m_index as a set of M with model m


model.const_oper_loadBalanceOfChemManu = Constraint(
    model.M, model.D, model.H, rule=const_oper_loadBalanceOfChemManu_rule
)


def const_oper_genCapacity_rule(m, n, i, d, h):
    return (
        m.p_gen[n, i, d, h] <= m.gen_c_gen_init[n, i] + m.c_gen[n, i]
    )  # generator capacity constraint


model.const_oper_genCapacity = Constraint(
    model.G, model.D, model.H, rule=const_oper_genCapacity_rule
)


def const_oper_genRampingUp_rule(m, n, i, d, h):
    if (d == 1) and (h == 1):
        return Constraint.Skip
    elif h == 1:
        return (
            m.p_gen[n, i, d, h] - m.p_gen[n, i, d - 1, m.H.last()] <= m.gen_r_ramp[n, i]
        )
    else:
        return (
            m.p_gen[n, i, d, h] - m.p_gen[n, i, d, h - 1] <= m.gen_r_ramp[n, i]
        )  # ramping constraints for generators


model.const_oper_genRampingUp = Constraint(
    model.G, model.D, model.H, rule=const_oper_genRampingUp_rule
)


def const_oper_genRampingDown_rule(m, n, i, d, h):
    if (d == 1) and (h == 1):
        return Constraint.Skip
    elif h == 1:
        return (
            m.p_gen[n, i, d, h] - m.p_gen[n, i, d - 1, m.H.last()]
            >= -m.gen_r_ramp[n, i]
        )
    else:
        return (
            m.p_gen[n, i, d, h] - m.p_gen[n, i, d, h - 1] >= -m.gen_r_ramp[n, i]
        )  # ramping constraints for generators


model.const_oper_genRampingDown = Constraint(
    model.G, model.D, model.H, rule=const_oper_genRampingDown_rule
)


# transimission constraints
def const_oper_transCapacityUpper_rule(m, n, n_prime, d, h):
    return (
        BaseMVA / m.line_x[n, n_prime] * (m.theta[n, d, h] - m.theta[n_prime, d, h])
        <= m.line_c_trans_init[n, n_prime] + m.c_trans[n, n_prime]
    )  # transmission capacity


model.const_oper_transCapacityUpper = Constraint(
    model.L, model.D, model.H, rule=const_oper_transCapacityUpper_rule
)


# storage constraints
def const_stor_storageLevel_rule(m, n, d, h):
    if (d == 1) and (h == 1):  # initial level should be designated
        return (
            m.p_stor_level[n, d, h]
            == m.stor_p_stor_level_init[n]
            + m.eta_charge * m.p_stor_charge[n, d, h]
            + m.eta_discharge * m.p_stor_discharge[n, d, h]
        )
    elif (
        h == 1
    ):  # first hour of the day (roll over from the last hour of the previous day)
        return m.p_stor_level[n, d, 1] == m.p_stor_level[n, d - 1, m.H.last()]
    else:
        return (
            m.p_stor_level[n, d, h]
            == m.p_stor_level[n, d, h - 1]
            + m.eta_charge * m.p_stor_charge[n, d, h]
            + m.eta_discharge * m.p_stor_discharge[n, d, h]
        )


model.const_stor_storageLevel = Constraint(
    model.N, model.D, model.H, rule=const_stor_storageLevel_rule
)


def const_stor_storageCapacity_rule(m, n, d, h):
    return m.p_stor_level[n, d, h] <= m.c_stor_init[n] + m.c_stor[n]


model.const_stor_storageCapacity = Constraint(
    model.N, model.D, model.H, rule=const_stor_storageCapacity_rule
)


# investment constraints
def const_invest_genCapacity_rule(m, n, i):
    return (
        m.c_gen[n, i] <= m.c_gen_max[n, i]
    )  # Maximum capacity for each generator type at each bus


model.const_invest_genCapacity = Constraint(model.G, rule=const_invest_genCapacity_rule)


def const_invest_transCapacity_rule(m, n, n_prime):
    return (
        m.c_trans[n, n_prime] <= m.c_trans_max[n, n_prime]
    )  # Maximum capacity for each transmission line between buses


model.const_invest_transCapacity = Constraint(
    model.L, rule=const_invest_transCapacity_rule
)


def const_invest_storCapacity_rule(m, n):
    return (
        m.c_stor[n] <= m.c_stor_max[n]
    )  # Maximum capacity for each storage at each bus


model.const_invest_storCapacity = Constraint(
    model.N, rule=const_invest_storCapacity_rule
)

# lower value bound for investment variables are determined by the domain of the variables, which is NonNegativeReals.
# model.const_invest_storCapacity.pprint()


# ----------------------------------------------------
# Objective Function
# ----------------------------------------------------


def obj_rule(m):
    # CAPEX
    capex_gen = sum(m.obj_gamma_gen[i] * m.c_gen[n, i] for (n, i) in m.G)
    capex_trans = sum(
        m.obj_gamma_trans * m.line_mile[n, n_prime] for (n, n_prime) in m.L
    )  # capex_trans is calculated by $/mile
    capex_stor = sum(m.obj_gamma_stor * m.c_stor[n] for n in m.N)
    capex = m.obj_phi * (capex_gen + capex_trans + capex_stor)

    # OPEX
    opex_gen = sum(
        m.obj_pi_gen[i] * m.p_gen[n, i, d, h]
        for (n, i) in m.G
        for d in m.D
        for h in m.H
    )
    opex_trans = sum(
        m.obj_pi_trans[n, n_prime]
        * abs(
            BaseMVA / m.line_x[n, n_prime] * (m.theta[n, d, h] - m.theta[n_prime, d, h])
        )
        for (n, n_prime) in m.L
        for d in m.D
        for h in m.H
    )
    opex_stor = sum(
        m.obj_pi_stor * m.p_stor_level[n, d, h] for n in m.N for d in m.D for h in m.H
    )
    opex_curt = sum(
        m.obj_pi_curt * m.curt[n, d, h] for n in m.N for d in m.D for h in m.H
    )
    opex = opex_gen + opex_trans + opex_stor + opex_curt

    return capex + opex


model.Objective = Objective(rule=obj_rule, sense=minimize)
# model.Objective.pprint()
