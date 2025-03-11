import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

# Record the start time
start_time = time.time()


# Load the dataset
TotalCase = 100
file_path = f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_{TotalCase}_data_CSTR.csv"
data = pd.read_csv(file_path)

# Prepare the input and output data
X = data[["production_target"]]
y = data["inlet_flow_rate"]
print(f"X: {X}")
print(f"y: {y}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# Scaler needed?

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 and 50 neurons
    activation="relu",  # Activation function: 'relu' (others: 'tanh', 'logistic')
    solver="adam",  # Optimizer: 'adam' (others: 'sgd', 'lbfgs')
    alpha=0.01,  # L2 regularization term (higher = stronger regularization)
    learning_rate="adaptive",  # Learning rate strategy
    max_iter=1000,  # Number of epochs
    random_state=42,
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Record the end time
end_time = time.time()

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

# Save the trained model
model_path = "/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/ann_model.pkl"
joblib.dump(mlp, model_path)
print(f"Model saved to {model_path}")

plt.scatter(X_test, y_test, label="Actual Data", color="blue", alpha=0.6)
plt.scatter(X_test, y_pred, label="Predicted Data", color="red", alpha=0.6)
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("MLP Regressor Predictions")
plt.show()
