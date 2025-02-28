import numpy as np
import matplotlib.pyplot as plt


# Define the matrix
matrix = np.array([[802, -400], [-400, 200]])

# Calculate the eigenvalues
eigenvalues = np.linalg.eigvals(matrix)
print(eigenvalues)
