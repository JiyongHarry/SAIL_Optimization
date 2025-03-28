import numpy as np
import matplotlib.pyplot as plt


def C_M(X, T, M):
    """Compute the truncated series solution C_M(X, T) with M terms."""
    C_approx = np.zeros_like(X)
    for n in range(1, M + 1, 2):  # Only odd terms
        C_approx += (
            (4 / (n * np.pi)) * np.sin(n * np.pi * X) * np.exp(-(n**2) * np.pi**2 * T)
        )
    return C_approx


# Define parameters
X = np.linspace(0, 1, 100)  # Spatial domain
T_values = [0, 0.001, 0.02, 0.05, 0.1, 0.2, 1.0]  # Given time values


# Approximate M selection based on error tolerance (simplified approach)
def find_M(T, epsilon_tol=0.01):
    M = 1
    while (4 / ((M + 2) * np.pi)) * np.exp(
        -((M + 2) ** 2) * np.pi**2 * T
    ) > epsilon_tol:
        M += 2  # Only odd values of M are considered
    return M


for T in T_values:
    M = find_M(T)
    print(f"For T={T}, M={M}")

# Plot solutions for different T values
plt.figure(figsize=(10, 6))
for T in T_values:
    M = find_M(T)  # Determine M for given T
    C_approx = C_M(X, T, M)
    plt.plot(X, C_approx, label=f"T={T}, M={M}")

plt.xlabel("X")
plt.ylabel("C(X, T)")
plt.title("Concentration Profile for Different T Values")
plt.legend()
plt.show()

# Special case: M = 1 for all T
plt.figure(figsize=(10, 6))
for T in T_values:
    C_approx = C_M(X, T, M=1)
    plt.plot(X, C_approx, label=f"T={T}, M=1")

plt.xlabel("X")
plt.ylabel("C_M=1(X, T)")
plt.title("Concentration Profile with M=1")
plt.legend()
plt.show()
