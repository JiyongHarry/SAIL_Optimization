import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 1)**4

def df(x):
    return 4 * (x - 1)**3

def ddf(x):
    return 12 * (x - 1)**2

def newtons_method(x0, tol=1e-6, max_iter=100):
    x = x0
    iterations = []
    for i in range(max_iter):
        fx = df(x)
        fxx = ddf(x)
        if abs(fx) < tol:
            break
        x_new = x - fx / fxx
        iterations.append((i+1, x, fx, x_new))
        x = x_new
    return x, iterations

def secant_method(xp, xq, tol=1e-6, max_iter=100):
    iterations = []
    for i in range(max_iter):
        f_xp = df(xp)
        f_xq = df(xq)
        if abs(f_xq) < tol:
            break
        x_new = xq - f_xq * (xq - xp) / (f_xq - f_xp)
        iterations.append((i+1, xp, xq, f_xp, f_xq, x_new))
        xp, xq = xq, x_new
    return xq, iterations

# Part (a): Newton's method
x0 = -1.0
x_min_newton, newton_iterations = newtons_method(x0)
print("Newton's Method:")
for it in newton_iterations:
    print(f"Iter {it[0]}: x = {it[1]:.6f}, df(x) = {it[2]:.6f}, x_new = {it[3]:.6f}")
print(f"Minimum found at x = {x_min_newton:.6f}\n")

# Part (b): Secant method
xp, xq = -2.0, 2.0  # Initial guesses
x_min_secant, secant_iterations = secant_method(xp, xq)
print("Secant Method:")
for it in secant_iterations:
    print(f"Iter {it[0]}: xp = {it[1]:.6f}, xq = {it[2]:.6f}, df(xp) = {it[3]:.6f}, df(xq) = {it[4]:.6f}, x_new = {it[5]:.6f}")
print(f"Minimum found at x = {x_min_secant:.6f}")

# Plot function
gx = np.linspace(-10, 10, 400)
gy = f(gx)
plt.plot(gx, gy, label='f(x) = (x-1)^4')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of (x-1)^4')
plt.legend()
plt.grid()
plt.show()