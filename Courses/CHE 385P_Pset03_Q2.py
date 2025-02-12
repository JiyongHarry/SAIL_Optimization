import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x1, x2):
    return (x1**2 + (x2 + 1)**2) * (x1**2 + (x2 - 1)**2)

# Generate grid points
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)
F = f(X1, X2)

# Plot the level curves (contour plot)
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, F, levels=200, cmap='viridis')
plt.colorbar(contours, label='f(x)')

# Mark the given points with labels
critical_points = np.array([[0, 0], [0, 1], [0, -1], [1, 1]])
labels = ['(a)', '(b)', '(c)', '(d)']
for point, label in zip(critical_points, labels):
    plt.scatter(point[0], point[1], color='red', marker='o')
    plt.text(point[0] + 0.1, point[1] + 0.1, label, fontsize=12, color='red')

# Labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Level Curves of $f(x)$')
plt.legend()
plt.grid()
plt.show()
