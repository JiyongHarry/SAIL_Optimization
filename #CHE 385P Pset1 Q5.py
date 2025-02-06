#CHE 385P Pset1 Q5

import numpy as np
import matplotlib.pyplot as plt

# Define x and y ranges
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(1, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Objective function f(x, y) = e^x + y
F = np.exp(X) + Y

# Constraint: e^x <= y (feasible region is where y >= e^x)
constraint_y = np.exp(x_vals)

# Plot contours of the objective function
plt.figure(figsize=(6,6))
contour = plt.contour(X, Y, F, levels=np.linspace(F.min(), F.max(), 10), cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# Shade feasible region
plt.fill_between(x_vals, constraint_y, 4, where=(constraint_y <= 4), color='lightblue', alpha=0.5, label="Feasible Region")

# Plot constraint boundary e^x = y
plt.plot(x_vals, constraint_y, 'r--', label=r"$y = e^x$")

# Plot bounds
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.axhline(1, color='black', linestyle='--', linewidth=1)
plt.axhline(4, color='black', linestyle='--', linewidth=1)

# Labels and legend
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Original program: NLP")
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# Define phi and y ranges
phi_vals = np.linspace(1, np.e, 100)
y_vals = np.linspace(1, 4, 100)
PHI, Y = np.meshgrid(phi_vals, y_vals)

# Objective function f(phi, y) = phi + y
F_LP = PHI + Y

# Constraint: phi <= y (feasible region is where y >= phi)
constraint_y = phi_vals  # y = phi

# Plot contours of the objective function
plt.figure(figsize=(6,6))
contour = plt.contour(PHI, Y, F_LP, levels=np.linspace(F_LP.min(), F_LP.max(), 10), cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# Shade feasible region
plt.fill_between(phi_vals, constraint_y, 4, where=(constraint_y <= 4), color='lightblue', alpha=0.5, label="Feasible Region")

# Plot constraint boundary phi = y
plt.plot(phi_vals, constraint_y, 'r--', label=r"$y = \phi$")

# Plot bounds
plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.axvline(np.e, color='black', linestyle='--', linewidth=1)
plt.axhline(1, color='black', linestyle='--', linewidth=1)
plt.axhline(4, color='black', linestyle='--', linewidth=1)

# Labels and legend
plt.xlabel(r"$\phi$")
plt.ylabel("$y$")
plt.title("Transformed program: LP")
plt.legend()
plt.grid(True)

# Show plot
plt.show()