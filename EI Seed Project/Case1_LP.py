# Basic imports
from pyomo.environ import *
import pandas as pd

# --- Load bus data from CSV ---
df_bus = pd.read_csv("EI Seed Project/TEXAS 123-BT params (bus).csv")
df_gen = pd.read_csv("EI Seed Project/TEXAS 123-BT params (generation).csv")
df_line = pd.read_csv("EI Seed Project/TEXAS 123-BT params (line).csv")

# --- Model ---
model = ConcreteModel()

# --- Sets ---
# Global parameters for time horizons
time_yearsHorizon = 10  # Number of years in the planning horizon
time_daysHorizon = 365  # Number of days in the planning horizon
time_hoursHorizon = 24  # Number of hours in a day


model.N = Set(initialize=df_bus["Bus Number"].unique())  # Set of buses
model.G = Set(
    initialize=df_gen["Gen Number"].unique()
)  # Set of generators (Genetartor#)
model.L = Set(initialize=df_line["line_num"].unique())  # Set of lines
model.D = Set(initialize=range(1, time_daysHorizon + 1))  # Set of days
model.H = Set(initialize=range(1, time_hoursHorizon + 1))  # Set of hours
model.I_gen = Set(
    initialize=df_gen["Fuel type"].unique()
)  # Set of generator technologies
model.I_gen_TH = Set(
    initialize=df_gen[
        df_gen["Fuel type"].isin(["Natura Gas", "Coal", "Nuclear", "Hydro"])
    ]["Fuel type"].unique()
)  # Set of thermal generator technologies
# Hydro is included as a parameter from the reference, but should not be considered as investment variable.
model.I_gen_RN = Set(
    initialize=df_gen[df_gen["Fuel type"].isin(["Wind", "Solar"])]["Fuel type"].unique()
)  # Set of renewable generators
model.C = Set(initialize=range(1, 6))  # Regional sets of data centers load zone (TBD)
model.M = Set(
    initialize=range(1, 6)
)  # Regional sets of chemical manufacturing load zone (TBD)

# --- Parameters ---
# ---- Read from TEXAS 123-BT params.csv files in the working direct. ----
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

model.bus_name = Param(model.N, initialize=bus_name_dict, within=Any)
model.bus_latitude = Param(
    model.N, initialize=bus_latitude_dict
)  # Q: Should I assign domain for each params?
model.bus_longitude = Param(model.N, initialize=bus_longitude_dict)
model.bus_genBool = Param(model.N, initialize=bus_genBool_dict)
model.bus_nominalVoltage = Param(model.N, initialize=bus_nominalVoltage_dict)
model.bus_weatherZone = Param(
    model.N, initialize=bus_weatherZone_dict, within=Any
)  # within=Any: this allows for any type of data, including strings or integers. default is Real, which is a floating-point number.

# ----- Generator Parameters ----
gen_toBusNumber_dict = {
    row["Gen Number"]: row["Bus Number"] for _, row in df_gen.iterrows()
}
gen_pMax_dict = {row["Gen Number"]: row["Pmax (MW)"] for _, row in df_gen.iterrows()}
gen_pMin_dict = {row["Gen Number"]: row["Pmin (MW)"] for _, row in df_gen.iterrows()}
gen_qMax_dict = {row["Gen Number"]: row["Qmax (MVar)"] for _, row in df_gen.iterrows()}
gen_qMin_dict = {row["Gen Number"]: row["Qmin (MVar)"] for _, row in df_gen.iterrows()}
gen_fuelType_dict = {
    row["Gen Number"]: row["Fuel type"] for _, row in df_gen.iterrows()
}
gen_c0_dict = {row["Gen Number"]: row["C0($/MWh)"] for _, row in df_gen.iterrows()}
gen_c1_dict = {row["Gen Number"]: row["C1($/MWh)"] for _, row in df_gen.iterrows()}
gen_cSu_dict = {row["Gen Number"]: row["Csu($)"] for _, row in df_gen.iterrows()}
gen_rampingRate_dict = {
    row["Gen Number"]: row["Ramping Rate(MW/min)"] for _, row in df_gen.iterrows()
}

model.gen_toBusNumber = Param(model.G, initialize=gen_toBusNumber_dict)
model.gen_pMax = Param(model.G, initialize=gen_pMax_dict)
model.gen_pMin = Param(model.G, initialize=gen_pMin_dict)
model.gen_qMax = Param(model.G, initialize=gen_qMax_dict)
model.gen_qMin = Param(model.G, initialize=gen_qMin_dict)
model.gen_fuelType = Param(model.G, initialize=gen_fuelType_dict, within=Any)
model.gen_c0 = Param(model.G, initialize=gen_c0_dict)
model.gen_c1 = Param(model.G, initialize=gen_c1_dict)
model.gen_cSu = Param(model.G, initialize=gen_cSu_dict)
model.gen_rampingRate = Param(model.G, initialize=gen_rampingRate_dict)

# ----- Line Parameters ----
line_fromBusNumber_dict = {
    row["line_num"]: row["From Bus Number"] for _, row in df_line.iterrows()
}
line_toBusNumber_dict = {
    row["line_num"]: row["To Bus Number"] for _, row in df_line.iterrows()
}
line_r_dict = {row["line_num"]: row["R, pu"] for _, row in df_line.iterrows()}
line_x_dict = {row["line_num"]: row["X, pu"] for _, row in df_line.iterrows()}
line_b_dict = {row["line_num"]: row["B, pu"] for _, row in df_line.iterrows()}
line_capactiy_dict = {
    row["line_num"]: row["Capacity (MW)"] for _, row in df_line.iterrows()
}
line_fromBusLatitude_dict = {
    row["line_num"]: row["From Bus Latitude"] for _, row in df_line.iterrows()
}
line_fromBusLongitude_dict = {
    row["line_num"]: row["From Bus Longitude"] for _, row in df_line.iterrows()
}
line_toBusLatitude_dict = {
    row["line_num"]: row["To Bus Latitude"] for _, row in df_line.iterrows()
}
line_toBusLongitude_dict = {
    row["line_num"]: row["To Bus Longitude"] for _, row in df_line.iterrows()
}
line_length_dict = {
    row["line_num"]: row["Length (Mile)"] for _, row in df_line.iterrows()
}

model.line_fromBusNumber = Param(model.L, initialize=line_fromBusNumber_dict)
model.line_toBusNumber = Param(model.L, initialize=line_toBusNumber_dict)
model.line_r = Param(model.L, initialize=line_r_dict)
model.line_x = Param(model.L, initialize=line_x_dict)
model.line_b = Param(model.L, initialize=line_b_dict)
model.line_capactiy = Param(model.L, initialize=line_capactiy_dict)
model.line_fromBusLatitude = Param(model.L, initialize=line_fromBusLatitude_dict)
model.line_fromBusLongitude = Param(model.L, initialize=line_fromBusLongitude_dict)
model.line_toBusLatitude = Param(model.L, initialize=line_toBusLatitude_dict)
model.line_toBusLongitude = Param(model.L, initialize=line_toBusLongitude_dict)
model.line_length = Param(model.L, initialize=line_length_dict)


# --- Verification Prints to check inputs ---
print("Bus set N:", list(model.N.data()))
for n in list(model.N.data())[: len(model.N)]:

    print(
        f"Bus {n}:",
        "| Bus name:",
        model.bus_name[n],
        "| Lat:",
        model.bus_latitude[n],
        "| Long:",
        model.bus_longitude[n],
        "| Gen bus:",
        model.bus_genBool[n],
        "| Voltage:",
        model.bus_nominalVoltage[n],
        "| Weather zone:",
        model.bus_weatherZone[n],
    )

print("\nGenerator set G:", list(model.G.data()))
for g in list(model.G.data())[: len(model.G)]:
    print(
        f"Generator {g}:",
        "| To bus number:",
        model.gen_toBusNumber[g],
        "| Pmax:",
        model.gen_pMax[g],
        "| Pmin:",
        model.gen_pMin[g],
        "| Qmax:",
        model.gen_qMax[g],
        "| Qmin:",
        model.gen_qMin[g],
        "| Fuel type:",
        model.gen_fuelType[g],
        "| C0:",
        model.gen_c0[g],
        "| C1:",
        model.gen_c1[g],
        "| Csu:",
        model.gen_cSu[g],
        "| Ramping rate:",
        model.gen_rampingRate[g],
    )

print("\nLine set L:", list(model.L.data()))
for l in list(model.L.data())[: len(model.L)]:
    print(
        f"Line {l}:",
        "| From Bus:",
        model.line_fromBusNumber[l],
        "| To Bus:",
        model.line_toBusNumber[l],
        "| R:",
        model.line_r[l],
        "| X:",
        model.line_x[l],
        "| B:",
        model.line_b[l],
        "| Capacity:",
        model.line_capactiy[l],
        "| From Lat:",
        model.line_fromBusLatitude[l],
        "| From Long:",
        model.line_fromBusLongitude[l],
        "| To Lat:",
        model.line_toBusLatitude[l],
        "| To Long:",
        model.line_toBusLongitude[l],
        "| Length:",
        model.line_length[l],
    )

# --- Variables ---
# Example variable: x[n,d,h]
# model.x = Var(model.N, model.D, model.H, domain=NonNegativeReals)

model.p_gen = Var(
    model.N, model.I_gen, model.D, model.H, domain=NonNegativeReals
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


# --- Constraints ---
# operational constraints
def const_oper_energyBalance_rule(m, n, d, h):
    return (
        sum(m.p_gen[n, i, d, h] for i in m.I_gen)
        + m.p_discharge[n, d, h]
        - m.p_charge[n, d, h]
        - sum(
            m.B[n, n_prime] * (m.theta[n, d, h] - m.theta[n_prime, d, h])
            for n_prime in m.N_neighbor
            if (n, n_prime) in m.B
        )  # NEED CHECK
        == m.D_res[n, d, h]
        + m.p_chemManu[n, d, h]
        + m.p_dataCenter[n, d, h]
        - m.curt[n, d, h]
    )


def const_oper_loadBalanceOfDataCenter_rule(m, c, d, h):
    return sum(m.p_dataCenter[n, d, h] for n in m.N_c[c]) == m.D_dataCenter[c, d, h]

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
    return (
        m.B[n, n_prime] * (m.theta[n, d, h] - m.theta[n_prime, d, h])
        >= - (m.c_trans0[n, n_prime] + m.c_trans[n, n_prime])
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

#investment
def const_invest_genCapacity_rule(m, n, i):
    return m.c_gen[n, i] <= m.c_gen_max[n, i]  # Maximum capacity for each generator type at each bus

def const_invest_transCapacity_rule(m, n, n_prime):
    return m.c_trans[n, n_prime] <= m.c_trans_max[n, n_prime]  # Maximum capacity for each transmission line between buses

def const_invest_storCapacity_rule(m, n):
    return m.c_stor[n] <= m.c_stor_max[n]  # Maximum capacity for each storage at each bus


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

# --- Objective ---
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
