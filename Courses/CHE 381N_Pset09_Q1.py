import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Define the given function h(X)
def h(X):
    return 4 * X**3 - 7 * X**2 + 2 * X


# Define lambda_n
def lambda_n(n):
    return (n + 0.5) * np.pi


# Function to compute the integral term
def integral_term(n):
    integrand = lambda X: h(X) * np.sin(lambda_n(n) * X)
    result, _ = quad(integrand, 0, 1)
    return 2 * result


# Define C(X,Y)
def C(X, Y, beta=1.5):
    summation = 0
    for n in range(3):  # n = 0, 1, 2
        lam_n = lambda_n(n)
        term = (
            integral_term(n)
            * np.sin(lam_n * X)
            * np.sinh(lam_n * Y)
            / np.sinh(lam_n * beta)
        )
        summation += term
    return summation


# Define grid for X, Y
X_vals = np.linspace(0, 1, 100)
Y_vals = np.linspace(0, 1.5, 100)
X, Y = np.meshgrid(X_vals, Y_vals)

# Compute C(X,Y)
C_vals = np.vectorize(C)(X, Y)

# Plot contour of 1 + C(X, Y)
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, 1 + C_vals, levels=50, cmap="jet")
plt.colorbar(contour, label="1 + C(X,Y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Contour plot of 1 + C(X,Y)")
plt.show()
