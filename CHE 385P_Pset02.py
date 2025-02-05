import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return np.exp(x**2 + y**2) + x**2

# Create a grid of points for x and y
x_vals = np.linspace(0, 2, 500)
y_vals = np.linspace(0, 2, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate the function values over the grid
Z = f(X, Y)

# Apply the constraint y - x <= 0 (i.e., y <= x)
mask = Y <= X

# Mask out non-feasible region
Z[~mask] = np.nan  # Mask out non-feasible region

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the feasible region as contour
contour = plt.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label=r'$f(x, y) = \exp(x^2 + y^2) + x^2$')

# Plot boundary lines for constraints
plt.plot(x_vals, x_vals, 'r--', label=r'$y=x$ (constraint)')

# Labels and styling
plt.title(r"Contour of $f(x, y) = \exp(x^2 + y^2) + x^2$ with Feasible Region")
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 2)  # Set x-axis range from 0 to 2
plt.ylim(0, 2)  # Set y-axis range from 0 to 2
plt.legend()
plt.grid(alpha=0.3)

# Show the plot
plt.show()
