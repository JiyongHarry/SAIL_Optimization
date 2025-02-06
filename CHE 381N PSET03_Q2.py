import numpy as np
import matplotlib.pyplot as plt

# Define the range of T
T = np.linspace(0, 10, 300)

# Compute x(T) for different values of alpha
x_0 = np.sin(T)  # alpha = 0

alpha_1 = 1/2
A_1 = 2 / np.sqrt(4 - alpha_1**2)
omega_1 = np.sqrt(4 - alpha_1**2) / 2
decay_1 = alpha_1 / 2
x_1 = A_1 * np.exp(-decay_1 * T) * np.sin(omega_1 * T)  # alpha = 1/2

alpha_2 = 1
A_2 = 2 / np.sqrt(4 - alpha_2**2)
omega_2 = np.sqrt(4 - alpha_2**2) / 2
decay_2 = alpha_2 / 2
x_2 = A_2 * np.exp(-decay_2 * T) * np.sin(omega_2 * T)  # alpha = 1

x_3 = T * np.exp(-T)  # alpha = 2

alpha_4 = 4
beta_4 = np.sqrt(alpha_4**2 - 4)
x_4 = (1 / beta_4) * (np.exp(-alpha_4 * T / 2) * (np.exp(beta_4 * T / 2) - np.exp(-beta_4 * T / 2)))  # alpha = 4

# Define distinct colors
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Plot all functions together
plt.figure(figsize=(8, 5))
plt.plot(T, x_0, label=r'$\alpha = 0$', color=colors[0])
plt.plot(T, x_1, label=r'$\alpha = 1/2$', color=colors[1])
plt.plot(T, x_2, label=r'$\alpha = 1$', color=colors[2])
plt.plot(T, x_3, label=r'$\alpha = 2$', color=colors[3])
plt.plot(T, x_4, label=r'$\alpha = 4$', color=colors[4])

plt.xlabel(r'$T$')
plt.ylabel(r'$x(T)$')
plt.title(r'Comparison of $x(T)$ for Different $\alpha$ Values')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
