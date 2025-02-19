import numpy as np
import matplotlib.pyplot as plt

# Create a grid of points
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)

# Define the vector components
U = X  # x-component: x
V = Y  # y-component: y

plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field: v(x) = x e$_x$ + y e$_y$')
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Create a grid of points in the xy-plane
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)

# Define the vector field components: w(x,y) = (-y, x)
U = -Y   # x-component of the vector field
V = X    # y-component of the vector field

# Plot the vector field
plt.figure(figsize=(6,6))
plt.quiver(X, Y, U, V, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field: w(x) = x e$_y$ - y e$_x$')
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling for x and y axes
plt.show()
