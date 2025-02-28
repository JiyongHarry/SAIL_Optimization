import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Define the function
def f(x):
    x1, x2 = x
    return x1**2 - x1 + x2**2 - x2 - 4

# Define the gradient of f
def grad_f(x):
    x1, x2 = x
    return np.array([2*x1 - 1, 2*x2 - 1])

# Define the Hessian matrix for Newton's method
def hessian_f(x):
    return np.array([[2, 0], [0, 2]])

# Exact line search function (minimizing f along a direction d from x)
def exact_line_search(x, d):
    phi = lambda alpha: f(x + alpha * d)
    res = minimize_scalar(phi)
    return res.x if res.success else 1.0  # Default to 1 if failed

# Newton's Method
def newtons_method(x0, tol=1e-6, max_iter=50):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        hess_inv = np.linalg.inv(hessian_f(x))  # Hessian is constant
        d = -hess_inv @ grad  # Newton direction
        alpha = exact_line_search(x, d)
        x_new = x + alpha * d
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {f(x_new)}")
        if np.linalg.norm(grad_f(x_new)) < tol:
            break
        x = x_new
    return x

# Quasi-Newton Method (BFGS)
def quasi_newton_bfgs(x0, tol=1e-6, max_iter=50):
    x = x0
    B = np.eye(2)  # Initial Hessian approximation
    for i in range(max_iter):
        grad = grad_f(x)
        d = -np.linalg.inv(B) @ grad  # Search direction
        alpha = exact_line_search(x, d)
        x_new = x + alpha * d
        s = x_new - x
        y = grad_f(x_new) - grad
        if np.dot(y, s) > 0:  # BFGS update condition
            B += np.outer(y, y) / np.dot(y, s) - np.outer(B @ s, B @ s) / np.dot(s, B @ s)
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {f(x_new)}")
        if np.linalg.norm(grad_f(x_new)) < tol:
            break
        x = x_new
    return x

# Conjugate Gradient Method
def conjugate_gradient(x0, tol=1e-6, max_iter=50):
    x = x0
    grad = grad_f(x)
    d = -grad  # Initial direction
    for i in range(max_iter):
        alpha = exact_line_search(x, d)
        x_new = x + alpha * d
        grad_new = grad_f(x_new)
        beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
        d = -grad_new + beta * d
        print(f"Iteration {i+1}: x = {x_new}, f(x) = {f(x_new)}")
        if np.linalg.norm(grad_new) < tol:
            break
        x, grad = x_new, grad_new
    return x

# Plot f(x) surface
x1_vals = np.linspace(-2, 2, 100)
x2_vals = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f([X1, X2])

plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Contour plot of f(x)')
plt.show()

# Initial point
x0 = np.array([1.0, 1.0])

print("\nNewton's Method:")
newtons_method(x0)

print("\nQuasi-Newton (BFGS) Method:")
quasi_newton_bfgs(x0)

print("\nConjugate Gradient Method:")
conjugate_gradient(x0)
