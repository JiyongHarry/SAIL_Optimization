import numpy as np
import matplotlib.pyplot as plt


# Define the full function V(Z, T)
def V(Z, T, N_terms=100):
    result = Z.copy()  # steady-state part
    for n in range(1, N_terms + 1):
        term = (
            (2 * (-1) ** n / (n * np.pi))
            * np.exp(-(n**2) * np.pi**2 * T)
            * np.sin(n * np.pi * Z)
        )
        result += term
    return result


# Set up Z and T values
Z = np.linspace(0, 1, 500)
T_values_focused = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 100]

# Plot the function for different T values
plt.figure(figsize=(10, 6))
for T in T_values_focused:
    V_values = V(Z, T)
    plt.plot(Z, V_values, label=f"T={T}")

plt.title("Plot of V(Z, T)")
plt.xlabel("Z")
plt.ylabel("V(Z, T)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
