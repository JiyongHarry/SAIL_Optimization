import math

def f_prime(z):
    return 0.1 * z + math.tanh(z)

def f_double_prime(z):
    return 0.1 + 1 / (math.cosh(z) ** 2)

def newton_method(z0, tol=1e-6, max_iter=100):
    z = z0
    for i in range(max_iter):
        f1 = f_prime(z)
        f2 = f_double_prime(z)
        
        print(f"Iteration {i+1}: z = {z}, f'(z) = {f1}, f''(z) = {f2}")
        
        if abs(f1) < tol:
            return z, i+1  # Converged
        
        z_new = z - f1 / f2
        
        if abs(z_new - z) < tol:
            return z_new, i+1  # Converged
        
        z = z_new
    
    return z, max_iter  # Did not converge within max_iter

# Run Newton's method from z = 10
z_opt, iterations = newton_method(10)
print(f"Optimal solution: z = {z_opt}, found in {iterations} iterations")

import numpy as np
import matplotlib.pyplot as plt

def f(z):
    return 0.05 * z**2 + np.log(np.cosh(z))

# Create an array of z values from -20 to 20
z_vals = np.linspace(-20, 20, 400)
f_vals = f(z_vals)

plt.figure(figsize=(8, 6))
plt.plot(z_vals, f_vals, label=r"$f(z)=0.05z^2+\ln(\cosh(z))$")
plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("Plot of f(z)")
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def f_prime(z):
    return 0.1 * z + np.tanh(z)

# Create an array of z values from -20 to 20
z_vals = np.linspace(-20, 20, 400)
fprime_vals = f_prime(z_vals)

plt.figure(figsize=(8, 6))
plt.plot(z_vals, fprime_vals, label=r"$f'(z)=0.1z+\tanh(z)$")
plt.xlabel("z")
plt.ylabel("f'(z)")
plt.title("Plot of f'(z) vs z")
plt.legend()
plt.grid(True)
plt.show()

