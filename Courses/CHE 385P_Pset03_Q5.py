import math

# Given parameters
Pc = 20
A = 250
mu = 20
M = 75
c = 0.01
a = 3.643
beta = 2.680
b = 3.2e-8

# Compute constant C
C = beta * Pc * A**2 / (mu * M**2 * c)

def f(x):
    """Objective function f(x) = -C * x * exp(-a*x + b)."""
    return -C * x * math.exp(-a*x + b)

def fprime(x):
    """First derivative: f'(x) = -C * exp(-a*x+b) * (1 - a*x)."""
    return -C * math.exp(-a*x + b) * (1 - a*x)

def fdouble(x):
    """Second derivative: f''(x) = a * C * exp(-a*x+b) * (2 - a*x)."""
    return a * C * math.exp(-a*x + b) * (2 - a*x)

def newton_method(x0, tol=1e-6, max_iter=100):
    """
    Applies Newton's method to find a stationary point of f(x).
    Since we wish to minimize f(x), we are solving f'(x)=0.
    """
    x = x0
    for i in range(max_iter):
        fp = fprime(x)
        fpp = fdouble(x)
        if fpp == 0:
            print("Zero second derivative encountered; stopping iteration.")
            break
        dx = -fp / fpp
        x = x + dx
        print(f"Iteration {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
        if abs(dx) < tol:
            break
    return x

# Choose an initial guess inside the interval [0,1]
x0 = 0.5
x_candidate = newton_method(x0)

print("\nNewton's method converged to:")
print(f"  x = {x_candidate:.6f}")
print(f"  f(x) = {f(x_candidate):.6f}")

# Evaluate f(x) at the boundaries
f0 = f(0)
f1 = f(1)
print("\nFunction values at the boundaries:")
print(f"  f(0) = {f0:.6f}")
print(f"  f(1) = {f1:.6f}")

# Since the feasible set is [0,1], compare the candidate with the endpoints.
candidates = [0, 1]
if 0 <= x_candidate <= 1:
    candidates.append(x_candidate)

# Determine the global minimizer (the one with the smallest f(x))
optimal_x = min(candidates, key=lambda x: f(x))
print("\nOptimal solution:")
print(f"  x_c = {optimal_x:.6f}")
print(f"  Minimum f(x) = {f(optimal_x):.6f}")
