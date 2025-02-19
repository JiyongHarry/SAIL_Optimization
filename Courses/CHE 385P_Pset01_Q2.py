import numpy as np
import matplotlib.pyplot as plt

# Given parameters
VT = 10  # Total volume in m^3
P = 30  # Cost per square meter
R_min = 1
R_max = 10

# Feasible region boundaries (derived earlier)
def feasible_height(d):
    return 4 * VT / (np.pi * d**2)

def ratio_bounds(d, R):
    return R * d

# Generate d values for plotting
d_vals = np.linspace(0.93, 2.68, 500)  # Between d_min and d_max
h_vals = feasible_height(d_vals)

# Define the cost function
def cost_function(d, h):
    return P * (np.pi * d * h + np.pi * d**2 / 2)

# Generate grid for contour plotting
d_grid, h_grid = np.meshgrid(d_vals, np.linspace(0, 30, 500))
z_grid = cost_function(d_grid, h_grid)

# Mask for the feasible region
feasible_mask = (h_grid >= ratio_bounds(d_grid, R_min)) & (h_grid <= ratio_bounds(d_grid, R_max))
z_grid[~feasible_mask] = np.nan  # Mask out non-feasible region

# Plotting
plt.figure(figsize=(8, 6))

# Plot feasible region contour
contour = plt.contourf(d_grid, h_grid, z_grid, levels=25, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='Cost (USD)')

# Plot volume constraint curve
plt.plot(d_vals, h_vals, color='red', linewidth=3, label='Volume Constraint')

# Add bounds for height-to-diameter ratio
plt.plot(d_vals, ratio_bounds(d_vals, R_min), 'orange', linestyle='--', linewidth=2, label='h/D = 1 (Min)')
plt.plot(d_vals, ratio_bounds(d_vals, R_max), 'blue', linestyle='--', linewidth=2, label='h/D = 10 (Max)')

# Labels and styling
plt.title("Optimization Problem: Cylindrical Vessel Design")
plt.xlabel("Diameter (D) [m]")
plt.ylabel("Height (h) [m]")
plt.xlim(0, 3)  # Set d-axis range from 0 to 3
plt.legend()
plt.grid(alpha=0.3)
plt.show()
