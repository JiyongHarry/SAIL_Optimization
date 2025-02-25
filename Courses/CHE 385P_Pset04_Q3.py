import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + 3*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 6*x[1]])

def exact_line_search(x, d):
    """Find optimal step size alpha by minimizing f(x + alpha*d) analytically."""
    num = -np.dot(grad_f(x), d)
    denom = np.dot(d, grad_f(x + d))
    return num / denom if denom != 0 else 1.0

def steepest_descent(x0, tol=1e-3, max_iters=100):
    xk = x0
    trajectory = [xk]
    
    for _ in range(max_iters):
        
        print(f"Iteration {_}: xk = {xk}, f = {f(xk)}, norm(f') = {np.linalg.norm(grad_f(xk))}")

        grad = grad_f(xk)
        if np.linalg.norm(grad) < tol:
            break
            
        d = -grad
        alpha = 0.5 * exact_line_search(xk, d)
        xk = xk + alpha * d

        trajectory.append(xk)
    
    return np.array(trajectory)

# Initial point
x0 = np.array([4.0, 2.0])
trajectory = steepest_descent(x0)

# Plot contour and optimization path
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f([X1, X2])

plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, levels=20)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', label='Optimization Path')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Steepest Descent Path on Contour Plot')
plt.legend()
plt.grid()
plt.show()

# Alternative Step Size Strategy: Fixed Step Size
# Instead of computing alpha by exact line search, we can use an optimized fixed step size:
def steepest_descent_fixed_alpha(x0, alpha=0.2, tol=1e-3, max_iters=100):
    xk = x0
    trajectory = [xk]
    
    for _ in range(max_iters):
        grad = grad_f(xk)
        if np.linalg.norm(grad) < tol:
            break
        
        xk = xk - alpha * grad
        trajectory.append(xk)
    
    return np.array(trajectory)

trajectory_fixed = steepest_descent_fixed_alpha(x0)
print(f"Number of iterations with exact line search: {len(trajectory)}")
print(f"Number of iterations with fixed step size: {len(trajectory_fixed)}")

