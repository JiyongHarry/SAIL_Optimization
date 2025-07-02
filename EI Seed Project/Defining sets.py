# Basic imports
from pyomo.environ import *
import pandas as pd

# --- Load bus data from CSV ---
df_bus = pd.read_csv("EI Seed Project/TEXAS 123-BT params (bus).csv")
# --- Model ---
model = ConcreteModel()

# --- Sets ---
# model.N = Set(initialize=range(1, 124))  # Set of buses
# model.L = Set(initialize=range(1, 255))  # Set of lines
# model.D = Set(initialize=range(1, 366))  # Set of days 1 to 365
# model.H = Set(initialize=range(1, 25))   # Set of hours 1 to 24
# model.I^{gen} = Set(initialize=['Natura Gas', 'Coal', 'Hydro','Nuclear', 'Wind', 'Solar'])  # Set of generator type. Hydro is considered as a params from the ref. But no investment var.
# model.I^{gen_TH} = Set('Natura Gas', 'Coal','Nuclear') # Set of thermal generators. Hydro is omitted
# model.I^{gen_RN} = Set('Wind', 'Solar')  # Set of renewable generators
# model.C = Set(initailize=range(1,6)) # Regional sets of data centers load zone. TBD
# model.M = Set(initialize=range(1,6)) # Regional sets of chemical manufacturing load zone. TBD
model.N = Set(initialize=df_bus["Bus Number"].unique())  # Set of buses

# --- Parameters ---
bus_name_dict = {row["Bus Number"]: row["Bus Name"] for _, row in df_bus.iterrows()}
bus_latitude_dict = {
    row["Bus Number"]: row["Bus latitude"] for _, row in df_bus.iterrows()
}
bus_longitude_dict = {
    row["Bus Number"]: row["Bus longitude"] for _, row in df_bus.iterrows()
}
gen_bus_dict = {
    row["Bus Number"]: row["Gen bus/ Non-gen bus"] for _, row in df_bus.iterrows()
}
nominal_voltage_dict = {
    row["Bus Number"]: row["Nominal Voltage (KV)"] for _, row in df_bus.iterrows()
}
weather_zone_dict = {
    row["Bus Number"]: row["Weather Zone"] for _, row in df_bus.iterrows()
}

model.bus_name = Param(model.N, initialize=bus_name_dict, within=Any)
model.bus_latitude = Param(model.N, initialize=bus_latitude_dict)
model.bus_longitude = Param(model.N, initialize=bus_longitude_dict)
model.gen_bus = Param(model.N, initialize=gen_bus_dict)
model.nominal_voltage = Param(model.N, initialize=nominal_voltage_dict)
model.weather_zone = Param(model.N, initialize=weather_zone_dict, within=Any)


# --- Verification Prints ---
print("Bus set N:", list(model.N.data()))
print("First 5 bus names:")
for n in list(model.N.data())[:123]:

    print(
        f"Bus {n}:",
        model.bus_name[n],
        "| Lat:",
        model.bus_latitude[n],
        "| Long:",
        model.bus_longitude[n],
        "| Gen bus:",
        model.gen_bus[n],
        "| Voltage:",
        model.nominal_voltage[n],
        "| Weather zone:",
        model.weather_zone[n],
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
