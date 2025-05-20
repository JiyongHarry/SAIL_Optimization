import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 2D grid
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# Define the function:
# Smooth wave pattern + saddle behavior
Z = np.sin(3 * X) * np.cos(3 * Y) * np.exp(-0.3 * (X**2 + Y**2)) + 0.2 * X * Y

# Plot the surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="Spectral", edgecolor="none", alpha=0.95)

# Labels and title
ax.set_title(
    "Smooth Function with Local Minima, Maxima, and Saddle Points", fontsize=13
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt


# Create a 2D grid of points
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)

# Define the saddle function: f(x, y) = x^2 - y^2
Z = X**2 - Y**2

# Plotting the surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none", alpha=0.9)
ax.set_title("Saddle point of somewhere in objective fucntion", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

# Mark the saddle point
ax.scatter(0, 0, 0, color="k", s=50, label="Saddle Point")
ax.legend()

plt.tight_layout()
plt.show()
