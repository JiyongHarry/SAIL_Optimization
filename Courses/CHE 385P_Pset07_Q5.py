import numpy as np
import matplotlib.pyplot as plt


# Define the constraint function: (x1 - 1)^3 - x2^2 = 0
def constraint(x1):
    return np.sqrt((x1 - 1) ** 3)  # Take square root, only real solutions


# Generate x1 values in a reasonable range
x1_values = np.linspace(-10, 10, 400)
x2_values_positive = constraint(x1_values)
x2_values_negative = (
    -x2_values_positive
)  # Since x2^2 = (x1-1)^3, take both positive and negative branches

# Define the objective function contours
X1, X2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Z = X1**2 + X2**2  # Objective function

# Plot the results
plt.figure(figsize=(8, 8))
plt.contour(
    X1, X2, Z, levels=100, cmap="viridis", alpha=0.7
)  # Contour plot of the objective function
plt.plot(x1_values, x2_values_positive, "r", label=r"$(x_1 - 1)^3 - x_2^2 = 0$")
plt.plot(x1_values, x2_values_negative, "r")

# Find and highlight the optimal point (graphically)
optimal_x1 = 1  # Since minimizing distance from origin, it must be near (1,0)
optimal_x2 = 0
plt.scatter(optimal_x1, optimal_x2, color="red", marker="o", label="Optimal Point")

# Labels and legend
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Visualization of Constrained Optimization Problem")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend()
plt.grid(True)
plt.show()
