import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time


# Define the read_data function
def read_data(file_path):
    df_p = pd.read_excel(file_path, sheet_name="p_val")  # First tab for p_val_array
    df_u = pd.read_excel(file_path, sheet_name="u_val")  # Second tab for u_val_array
    print("Data from p_val sheet:")
    print(df_p)  # Print all rows
    print("Data from u_val sheet:")
    print(df_u)  # Print all rows
    p_val_array = df_p.values
    u_val_array = df_u.values
    return p_val_array, u_val_array


# Record the start time
start_time = time.time()

# Load the dataset
TotalCase = 10
file_path = f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_{TotalCase}_data_CSTR.xlsx"
p_val_array, u_val_array = read_data(file_path)

# Prepare the input and output data
X = p_val_array
y = u_val_array
print(f"X: {X}")
print(f"y: {y}")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=None  # fixed random_state when it's some integer
)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
print(f"y_train: {y_train}")
print(f"y_test: {y_test}")

# Scaler needed?

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 and 50 neurons
    activation="relu",  # Activation function: 'relu' (others: 'tanh', 'logistic')
    solver="adam",  # Optimizer: 'adam' (others: 'sgd', 'lbfgs')
    alpha=0.01,  # L2 regularization term (higher = stronger regularization)
    learning_rate="adaptive",  # Learning rate strategy
    max_iter=1000,  # Number of epochs
    random_state=None,  # fixed random_state when it's some integer
)

# Train the model
mlp.fit(X_train, y_train)

# Record the end time
end_time = time.time()

# Save the trained model
model_path = f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/MLPRegressor_model_{TotalCase}_case.pkl"
joblib.dump(mlp, model_path)

# Make predictions
y_pred = mlp.predict(X_test)

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2:.4f}")

# Plot
plt.figure(figsize=(10, 6))
for i in range(y_test.shape[0]):
    plt.plot(
        range(y_test.shape[1]),
        y_test[i, :],
        "bo-",
        label="Actual Data" if i == 0 else "",
        alpha=0.6,
    )
    plt.plot(
        range(y_test.shape[1]),
        y_pred[i, :],
        "r+-",
        label="Predicted Data" if i == 0 else "",
        alpha=0.6,
    )
plt.legend()
plt.xlabel("Column Index (j)")
plt.ylabel("Inlet Flow Rate [m3/s]")
plt.title("MLP Regressor Predictions (ND)")
plt.grid(True)
plt.show()
