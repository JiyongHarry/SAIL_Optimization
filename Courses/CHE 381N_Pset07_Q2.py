import numpy as np
import matplotlib.pyplot as plt

# Define X values
X = np.linspace(0, 1, 100)

# Compute C for different Pe values
Pe_values = [0, 1 / 4, 1, 4, 16]
C_values = {}

for Pe in Pe_values:
    if Pe == 0:
        C_values[Pe] = X  # C = X when Pe = 0
    else:
        C_values[Pe] = (np.exp(Pe * X) - 1) / (np.exp(Pe) - 1)

# Plot results
plt.figure(figsize=(8, 6))
for Pe, C in C_values.items():
    plt.plot(X, C, label=f"Pe = {Pe}")

plt.xlabel("X")
plt.ylabel("C")
plt.title("Concentration Profile for Different Peclet Numbers")
plt.legend()
plt.grid()
plt.show()
