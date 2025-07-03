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
model.N = Set(initialize=df_bus["Bus Number"].unique())  # Set of buses
model.G = Set(initialize=df_gen["Gen Number"].unique())  # Set of generators
model.L = Set(initialize=df_line["line_num"].unique())  # Set of lines
model.D = Set(initialize=range(1, 366))  # Set of days
model.H = Set(initialize=range(1, 25))  # Set of hours
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
# ---- Read from CSV and create dictionaries ----
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
